//! Experimental GDN/QSA/shared-expert projection overrides: upload unchanged
//! native GGUF blocks once, then attach them to the matching canonical
//! `WeightRef`. Port of `src/core/native_dense.cpp`.
//!
//! The bytes cross a seam, not a pointer: this core picks offsets and byte
//! counts and never names an address, so `NativeDense` runs unchanged against a
//! shim-backed arena in VRAM or a `Vec<u8>` in a test. One Q8_1 scratch region
//! is shared by every attached projection, exactly as the C++ shares one
//! `native_q8_1` — which is why every reader must execute on one ordered stream
//! and the object must outlive the graphs that reference it.

use std::collections::BTreeSet;
use std::path::Path;

use crate::native_mm::{native_mmvq_supported, native_mmvq_weight_bytes, native_q8_1_bytes};
use crate::weights::WeightTable;

/// The name prefix every layer-owned tensor carries.
const BLK: &str = "blk.";
const PLE_KEY: &str = "blk.1.ple_key.weight";

const SUFFIXES: [&str; 10] = [
    ".attn_qkv.weight",
    ".attn_gate.weight",
    ".ssm_out.weight",
    ".attn_q.weight",
    ".attn_k.weight",
    ".attn_v.weight",
    ".attn_output.weight",
    ".ffn_gate_shexp.weight",
    ".ffn_up_shexp.weight",
    ".ffn_down_shexp.weight",
];

/// One tensor as the GGUF header describes it, plus nothing the decoder needs:
/// eligibility and the span checks run on headers alone.
#[derive(Clone, Debug)]
pub struct NativeCandidate<'a> {
    pub name: String,
    pub ggml_type: i32,
    pub shape: Vec<u64>,
    /// payload offset relative to the shard's data start
    pub offset: u64,
    /// the tensor's own bytes, already read: the C++ `gguf.tensor_data(tensor)`
    pub data: &'a [u8],
}

/// A shard's header plus the metadata `load` arbitrates on. `payload_bytes` is
/// the whole mapped file — `data` slices of the tensors point into it.
#[derive(Clone, Debug)]
pub struct NativeShard<'a> {
    pub path: String,
    pub tensors: Vec<NativeCandidate<'a>>,
    pub data_start: u64,
    pub file_size: u64,
    /// What the shard's `general.architecture` key said. `Absent` is a
    /// continuation shard (no key of its own); `Failed` is the exact
    /// `check_architecture` message the C++ returns as its error.
    pub arch: Arch,
    /// `split.count`
    pub split_count: Option<u64>,
    /// `split.no`
    pub split_no: Option<u64>,
    /// `split.tensors.count`
    pub split_tensors: Option<u64>,
}

/// The three states of a shard's architecture key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arch {
    /// no `general.architecture` key — this shard is a continuation
    Absent,
    /// the key is there and `check_architecture` passed
    Validated,
    /// the key is there and `check_architecture` refused it, verbatim message
    Failed(String),
}

impl NativeShard<'_> {
    fn split(&self) -> Option<(u64, u64, u64)> {
        Some((self.split_count?, self.split_no?, self.split_tensors?))
    }
}

/// The device half. `alloc`/`free` mirror `cudaMalloc`/`cudaFree` in an offset
/// space this object owns, `upload` mirrors the `cudaMemcpy` of a tensor.
pub trait NativeUpload {
    /// Reserve `bytes` in the native space, returning their offset.
    fn alloc(&mut self, bytes: u64) -> Result<u64, String>;
    /// Deliver `bytes` at native offset `at`, device-visible on success.
    fn upload(&mut self, at: u64, bytes: &[u8]) -> Result<(), String>;
    /// Give back one region `alloc` returned.
    fn free(&mut self, at: u64, bytes: u64);
}

/// What `set_layer_range` recorded: `ALL` is every layer. Process-wide in the
/// C++ (a file-static read by the next `load`); here it is a value the caller
/// passes, which keeps two sessions from tripping over each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerRange {
    pub lo: i32,
    pub hi: i32,
}

impl LayerRange {
    pub const ALL: LayerRange = LayerRange { lo: -1, hi: -1 };

    /// The `in_range` of the C++: `std::atoi` of a non-numeric tail is 0, and
    /// the C++ compares that 0, so a name like `blk.x.attn_q.weight` is in
    /// range exactly when layer 0 is.
    fn holds(&self, name: &str) -> bool {
        if self.lo < 0 || !name.starts_with(BLK) {
            return true;
        }
        // atoi() of the tail after "blk." — leading digits only, 0 if none
        let l = layer_of(name).unwrap_or(0);
        l >= self.lo as i64 && l < self.hi as i64
    }
}

fn layer_of(name: &str) -> Option<i64> {
    name.strip_prefix(BLK).and_then(|rest| {
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits.parse::<i64>().ok()
    })
}

/// The eligible, supported, 2-D names `load` would serve natively, read from
/// the GGUF headers only — so the canonical arena can skip them.
pub fn served_names(shards: &[NativeShard<'_>], include_ple_key: bool) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for shard in shards {
        for t in &shard.tensors {
            if eligible(&t.name, t.ggml_type, include_ple_key)
                && native_mmvq_supported(t.ggml_type)
                && t.shape.len() == 2
            {
                out.insert(t.name.clone());
            }
        }
    }
    out
}

/// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized
/// (`--compat-bf16`) serves the PLE from that row, so the key comes out of the
/// skip set and `load` does not upload the GGUF key over it. A quantized row
/// leaves `skip` unchanged; a key absent from `skip` never opens the index.
pub fn keep_unquantized_ple_key(
    pack_dir: &Path,
    skip: &mut BTreeSet<String>,
) -> Result<(), String> {
    if !skip.contains(PLE_KEY) {
        return Ok(());
    }
    if WeightTable::index_code_bits(pack_dir, PLE_KEY)? == 0 {
        skip.remove(PLE_KEY);
    }
    Ok(())
}

fn eligible(name: &str, ggml_type: i32, include_ple_key: bool) -> bool {
    if !name.starts_with(BLK) {
        return false;
    }
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0
    // (UD-Q4_K_XL). Other keys retain the packed BF16 fallback.
    if name == PLE_KEY {
        return include_ple_key && matches!(ggml_type, 42 | 18 | 23 | 8);
    }
    SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// The payload byte count of a tensor from its type and shape, with the C++'s
/// three refusal reasons kept distinct: a geometry the decoder does not know
/// (or a shape too short to have an ne0), an extent that is zero or overflows,
/// and a byte count that overflows.
enum Spanned {
    Bytes(u64),
    Geometry,
    Extent,
    Overflow,
}

fn payload_span(ggml_type: i32, shape: &[u64]) -> Spanned {
    let geom = match (
        shape.first(),
        strata_artifact::gguf::block_geometry(ggml_type as u32),
    ) {
        (Some(_), Some(g)) => g,
        _ => return Spanned::Geometry,
    };
    let (elems, bytes) = (u64::from(geom.0), u64::from(geom.1));
    if !shape[0].is_multiple_of(elems) {
        return Spanned::Geometry;
    }
    let mut elements = 1u64;
    for &dim in shape {
        if dim == 0 || elements > u64::MAX / dim {
            return Spanned::Extent;
        }
        elements *= dim;
    }
    match (elements / elems).checked_mul(bytes) {
        Some(b) => Spanned::Bytes(b),
        None => Spanned::Overflow,
    }
}

/// What `load` decided, one per uploaded matrix, before any reference is
/// published (the C++ `Pending`).
#[derive(Clone, Debug)]
pub struct Pending {
    pub name: String,
    pub native_type: i32,
    pub bytes: u64,
    /// this matrix's offset in the `NativeUpload`'s space
    pub at: u64,
    /// the shared Q8_1 scratch's offset, identical for all of them
    pub scratch_at: u64,
}

/// Every region `load` allocated, so the owner can hand them back like the C++
/// destructor's `cudaFree` loop.
#[derive(Default, Debug)]
pub struct NativeDense {
    owned: Vec<(u64, u64)>,
    scratch: Option<(u64, u64)>,
    bytes: u64,
    count: usize,
}

impl NativeDense {
    pub fn weight_bytes(&self) -> u64 {
        self.bytes
    }
    pub fn tensor_count(&self) -> usize {
        self.count
    }
    pub fn scratch_region(&self) -> Option<(u64, u64)> {
        self.scratch
    }

    /// Attach every eligible native matrix in `shards` to `table`.
    ///
    /// `stage` bounds `blk.<l>.` tensors to one layer split's own layers (the
    /// stage holds its own projections, not the whole model's — the others keep
    /// `native_off == None`); `range` is the process-wide layer range of the
    /// C++ `set_layer_range`.
    pub fn load(
        &mut self,
        shards: &[NativeShard<'_>],
        table: &mut WeightTable,
        up: &mut dyn NativeUpload,
        include_ple_key: bool,
        stage: (i64, i64),
        range: LayerRange,
    ) -> Result<Vec<Pending>, String> {
        let (layer_lo, layer_hi) = stage;
        if self.scratch.is_some() || self.count > 0 {
            return Err("native dense: already loaded".into());
        }
        if shards.is_empty() {
            return Err("native dense: at least one GGUF shard is required".into());
        }
        // a `blk.<l>.` tensor of another stage's layers
        let outside = |name: &str| -> bool {
            if layer_hi < 0 || !name.starts_with(BLK) {
                return false;
            }
            // std::strtol of a non-numeric tail is 0, and the C++ compares that 0
            let l = layer_of(name).unwrap_or(0);
            l < layer_lo || l >= layer_hi
        };

        let mut pending: Vec<Pending> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut max_in = 0i32;
        let mut total = 0u64;
        let mut split_count = 0u64;
        let mut split_tensors = 0u64;
        let mut split_numbers: BTreeSet<u64> = BTreeSet::new();
        let mut have_architecture = false;

        for shard in shards {
            let (count, number, tensors) = match shard.split() {
                Some(s) => (Some(s.0), Some(s.1), Some(s.2)),
                None => (None, None, None),
            };
            // The C++ returns the architecture refusal verbatim, without the
            // "native dense: " prefix the catch adds to everything else.
            if let Arch::Failed(why) = &shard.arch {
                return Err(why.clone());
            }
            if shard.arch == Arch::Validated {
                have_architecture = true;
                if let (Some(c), Some(n), Some(t)) = (count, number, tensors) {
                    if n == 0 && c > 1 {
                        split_count = c;
                        split_tensors = t;
                    }
                }
            } else if !have_architecture
                || split_count == 0
                || count.is_none()
                || number.is_none()
                || tensors.is_none()
                || count != Some(split_count)
                || number == Some(0)
                || number.is_some_and(|n| n >= split_count)
                || tensors != Some(split_tensors)
            {
                return Err("native dense: additional shard must match the architecture-validated first shard's split metadata".into());
            }
            if let Some(n) = number {
                if !split_numbers.insert(n) {
                    return Err("native dense: duplicate split shard number".into());
                }
            }

            let mut offsets: Vec<u64> = shard.tensors.iter().map(|t| t.offset).collect();
            offsets.sort_unstable();
            if offsets.windows(2).any(|w| w[0] == w[1]) {
                return Err("native dense: tensor payload offsets overlap".into());
            }

            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            // A file smaller than its own data_start is refused here; the C++
            // could not observe one because its mmap-backed reader rejects such
            // a header at open.
            let payload = match shard.file_size.checked_sub(shard.data_start) {
                Some(p) => p,
                None => {
                    return Err(format!(
                        "native dense: truncated payload {}",
                        shard
                            .tensors
                            .first()
                            .map_or(String::new(), |t| t.name.clone())
                    ))
                }
            };
            for t in &shard.tensors {
                // the C++'s own message order, one refusal reason per shape
                let bytes = match payload_span(t.ggml_type, &t.shape) {
                    Spanned::Bytes(b) => b,
                    Spanned::Geometry => {
                        return Err(format!("native dense: invalid block geometry {}", t.name))
                    }
                    Spanned::Extent => {
                        return Err(format!("native dense: invalid tensor extent {}", t.name))
                    }
                    Spanned::Overflow => {
                        return Err(format!(
                            "native dense: tensor byte count overflow {}",
                            t.name
                        ))
                    }
                };
                if t.offset > payload || bytes > payload - t.offset {
                    return Err(format!("native dense: truncated payload {}", t.name));
                }
                let next = offsets.partition_point(|&o| o <= t.offset);
                if next < offsets.len() && bytes > offsets[next] - t.offset {
                    return Err(format!("native dense: overlapping payload {}", t.name));
                }
            }

            for t in &shard.tensors {
                if !eligible(&t.name, t.ggml_type, include_ple_key) || outside(&t.name) {
                    continue;
                }
                if !range.holds(&t.name) && !t.name.contains("ple") {
                    continue;
                }
                if !seen.insert(t.name.clone()) {
                    return Err(format!("native dense: duplicate tensor {}", t.name));
                }
                let Some(canonical) = table.get_mut(&t.name) else {
                    return Err(format!(
                        "native dense: tensor absent from canonical table: {}",
                        t.name
                    ));
                };
                if canonical.native_off.is_some() {
                    return Err("native dense: override already attached".into());
                }
                if !native_mmvq_supported(t.ggml_type) {
                    continue;
                }
                // #326: the pack keeps an unquantized (--compat-bf16) key, which
                // the PLE reads from the arena
                if t.name == PLE_KEY && !canonical.quantized() {
                    continue;
                }
                let ne0 = canonical.ne0;
                let ne1 = canonical.ne1;
                if !canonical.quantized()
                    || t.shape.len() != 2
                    || ne0 <= 0
                    || ne0 > i64::from(i32::MAX)
                    || ne1 <= 0
                    || ne1 > i64::from(i32::MAX)
                    || t.shape[0] != ne0 as u64
                    || t.shape[1] != ne1 as u64
                {
                    return Err(format!("native dense: incompatible matrix {}", t.name));
                }
                let bytes = native_mmvq_weight_bytes(t.ggml_type, ne0 as i32, ne1 as i32)
                    .map_err(|e| format!("native dense: {e}"))?;
                let at = up
                    .alloc(bytes)
                    .map_err(|why| format!("native dense upload {}: {why}", t.name))?;
                // the span checks above bounded this tensor's bytes against the
                // payload, so the only remaining failure is a shard read short
                if t.data.len() as u64 != bytes {
                    return Err(format!("native dense: truncated payload {}", t.name));
                }
                up.upload(at, t.data)
                    .map_err(|why| format!("native dense upload {}: {why}", t.name))?;
                max_in = max_in.max(ne0 as i32);
                total = total.wrapping_add(bytes);
                pending.push(Pending {
                    name: t.name.clone(),
                    native_type: t.ggml_type,
                    bytes,
                    at,
                    scratch_at: 0, // filled once the scratch exists
                });
            }
        }

        if pending.is_empty() {
            return Err("native dense: no supported GDN/QSA matrices in supplied shards".into());
        }
        // the engine's one-argument form: `native_q8_1_bytes(max_in, 1)`
        let scratch_bytes =
            native_q8_1_bytes(max_in, 1).map_err(|e| format!("native dense scratch: {e}"))?;
        let scratch_at = up
            .alloc(scratch_bytes)
            .map_err(|why| format!("native dense scratch: {why}"))?;

        // All checks and allocations finish before publishing any reference.
        for item in &mut pending {
            item.scratch_at = scratch_at;
            let canonical = table.get_mut(&item.name).unwrap();
            canonical.native_off = Some(item.at);
            canonical.native_type = item.native_type;
            canonical.native_q8_1 = Some(scratch_at);
            self.owned.push((item.at, item.bytes));
        }
        self.owned.push((scratch_at, scratch_bytes));
        self.scratch = Some((scratch_at, scratch_bytes));
        self.bytes = total;
        self.count = pending.len();
        Ok(pending)
    }

    /// The destructor's `cudaFree` loop, handed to the allocator that owns the
    /// space. Dropping without calling this leaks on purpose: a graph that is
    /// still alive may read the weights, and the C++ header says this object
    /// must outlive every graph that references it.
    pub fn free_all(mut self, up: &mut dyn NativeUpload) {
        for (at, bytes) in self.owned.drain(..) {
            up.free(at, bytes);
        }
        self.scratch = None;
        self.bytes = 0;
        self.count = 0;
    }
}
