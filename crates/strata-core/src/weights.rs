//! The dense-weight loader: `<pack_dir>/index.txt`, the arena placement it
//! describes, and the byte transforms between the pack's form and the engine's.
//! Port of `src/core/weights.cpp`.
//!
//! Like the conversation snapshot, this core stays pointer-free: an arena is a
//! byte count here and device offsets are what the caller handed out, so the
//! same code runs against the shim's real VRAM and against a host `Vec` in a
//! test. The bytes cross the boundary through the [`Upload`] trait, which is
//! where `cudaHostAlloc`/`cudaMemcpy`/`cudaDeviceSynchronize` live on a GPU box.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::native_mm::{f16_from_f32, f32_from_f16};

/// 8 MiB of SOURCE per staging round, as in the C++.
const CHUNK: u64 = 8 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightKind {
    /// quantized planes, or F16 already stored as 2 bytes: copy straight through
    Verbatim = 0,
    /// the pack holds the bf16 value promoted to f32: take the HIGH 16 bits (exact)
    Bf16InF32 = 1,
    F32 = 2,
    /// an f16 value promoted to f32: a real f32->f16 conversion, not a truncation
    F16InF32 = 3,
}

/// One row of `index.txt`, exactly the 19 fields `pack_index.py` writes
/// (kinds 4/5 are the raw-16 forms, remapped to kinds 1/3 with `raw16` set).
#[derive(Clone, Debug)]
pub struct IndexRow {
    pub name: String,
    pub file: i32,
    pub kind: WeightKind,
    pub raw16: bool,
    pub src_off: u64,
    pub src_bytes: u64,
    pub dst_off: u64,
    pub dst_bytes: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub code_bits: i32,
    pub code_bias: i32,
    pub group_elems: i32,
    pub codebook: i32,
    pub has_offset: i32,
    pub codes_bytes: u64,
    pub scales_bytes: u64,
    pub offset_bytes: u64,
    pub scales_fp16: bool,
    pub act_kind: i32,
}

/// Where this weight's bytes live, as an OFFSET into whatever arena the caller
/// placed it in (the C++ held the device pointer `dst_base + dst_off`; the base
/// is the caller's `arena_base` and never enters this core).
#[derive(Clone, Debug)]
pub struct WeightRef {
    /// `None` for a skipped row: metadata valid, `resident == false`, no bytes.
    pub arena_off: Option<u64>,
    pub bytes: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub elements: i64,
    pub kind: WeightKind,
    pub code_bits: i32,
    pub code_bias: i32,
    pub group_elems: i32,
    pub codebook_iq4nl: bool,
    pub has_offset: bool,
    pub codes_bytes: u64,
    pub scales_bytes: u64,
    pub offset_bytes: u64,
    pub src_off: u64,
    pub src_bytes: u64,
    pub file_id: i32,
    pub scales_fp16: bool,
    pub act_kind: i32,
    /// Optional native GGUF projection, owned by `NativeDense`: an offset into
    /// that object's allocations, the type, and the shared scratch's offset.
    pub native_off: Option<u64>,
    pub native_q8_1: Option<u64>,
    pub native_type: i32,
    pub resident: bool,
}

impl WeightRef {
    pub fn quantized(&self) -> bool {
        self.code_bits != 0
    }
    pub fn wants_q8k(&self) -> bool {
        self.act_kind == 1
    }
}

impl Default for WeightRef {
    fn default() -> Self {
        WeightRef {
            arena_off: None,
            bytes: 0,
            ne0: 0,
            ne1: 0,
            elements: 0,
            kind: WeightKind::Verbatim,
            code_bits: 0,
            code_bias: 0,
            group_elems: 0,
            codebook_iq4nl: false,
            has_offset: false,
            codes_bytes: 0,
            scales_bytes: 0,
            offset_bytes: 0,
            src_off: 0,
            src_bytes: 0,
            file_id: 0,
            scales_fp16: false,
            act_kind: 0,
            native_off: None,
            native_q8_1: None,
            native_type: -1,
            resident: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LoadReport {
    pub tensors: usize,
    pub arena_bytes: u64,
    /// tensors written as 16 bits out of a 32-bit container
    pub re_rounded: usize,
    /// what those conversions saved against the pack
    pub bytes_saved: u64,
}

/// The device side of a load. `upload` must deliver exactly `bytes.len()` bytes
/// at `arena_off` and only report success once they are visible to the device.
/// A failure is reported as the C++ reports it — `"cudaMemcpy failed for
/// <name>"`, the CUDA error string itself discarded — so the trait carries only
/// the pass/fail.
pub trait Upload {
    fn upload(&mut self, arena_off: u64, bytes: &[u8]) -> Result<(), String>;
}

/// `<pack_dir>/<id>.bin` — the four files an index may name.
pub fn pack_file_name(id: i32) -> Option<&'static str> {
    match id {
        0 => Some("dense.bin"),
        1 => Some("embd.bin"),
        2 => Some("experts.bin"),
        3 => Some("extra.bin"),
        _ => None,
    }
}

/// The whitespace tokens of a row. C++ `sscanf` with 19 conversions succeeds on
/// a row with 19 or more tokens as long as each numeric field scans; the one
/// divergence from C++ is a name over 255 characters, which `%255s` would
/// silently truncate — here it is a parse failure instead, because a tensor
/// registered under a truncated name is exactly the wrong-offset bug this
/// parser exists to refuse.
fn split_row(line: &str) -> (Vec<&str>, usize) {
    let fields: Vec<&str> = line.split_ascii_whitespace().collect();
    if fields.iter().any(|f| f.len() > 255) {
        return (fields, 0);
    }
    let mut n = 0usize; // conversions that succeeded, in order
    for (i, f) in fields.iter().enumerate() {
        if i == 0 {
            n = 1; // %255s always consumes
            continue;
        }
        if f.parse::<i128>().is_err() {
            break;
        }
        n = i + 1;
    }
    // sscanf stops after its last conversion; fields past it are never looked at,
    // so a well-formed row reports the format's own width, not its field count.
    (fields, n.min(19))
}

/// The first `count` fields as unsigned numbers; the numeric scans are already
/// known to have succeeded, this only converts. `%llu` on a negative token
/// wraps exactly like glibc, which i128-then-truncate reproduces.
fn num(fields: &[&str], i: usize) -> Result<u64, String> {
    fields[i].parse::<i128>().map(|v| v as u64).map_err(|_| {
        format!(
            "index.txt: field {i} of a row is not a number: {}",
            fields[i]
        )
    })
}

fn snum(fields: &[&str], i: usize) -> Result<i64, String> {
    fields[i].parse::<i64>().map_err(|_| {
        format!(
            "index.txt: field {i} of a row is not a number: {}",
            fields[i]
        )
    })
}

/// Parse the `# align %d pool %llu tensors %d` header line; None if it is a
/// comment that is not the header.
fn parse_header(line: &str) -> Option<(i32, u64)> {
    let mut it = line.split_ascii_whitespace();
    if it.next() != Some("#") || it.next() != Some("align") {
        return None;
    }
    let align = it.next()?.parse::<i32>().ok()?;
    if it.next() != Some("pool") {
        return None;
    }
    let pool = it.next()?.parse::<u64>().ok()?;
    if it.next() != Some("tensors") {
        return None;
    }
    let tensors = it.next()?.parse::<i32>().ok()?;
    // scanf("... tensors %d", &t) == 3 requires that the %d CONSUMED a number;
    // trailing whitespace is fine, a fourth token is not counted.
    let _ = tensors;
    Some((align, pool))
}

fn read_lines(path: &Path) -> Result<Vec<String>, ()> {
    let data = std::fs::read(path).map_err(|_| ())?;
    Ok(String::from_utf8_lossy(&data)
        .lines()
        .map(str::to_string)
        .collect())
}

#[derive(Clone, Debug)]
pub struct WeightTable {
    entries: Vec<(String, WeightRef)>,
    index: HashMap<String, usize>,
    report: LoadReport,
}

impl WeightTable {
    pub fn new() -> WeightTable {
        WeightTable {
            entries: Vec::new(),
            index: HashMap::new(),
            report: LoadReport::default(),
        }
    }

    pub fn find(&self, name: &str) -> Option<&WeightRef> {
        self.index.get(name).map(|&i| &self.entries[i].1)
    }

    /// NativeDense attaches its uploads here; the C++ reached `table_` through
    /// `friend class NativeDense`, this is the same access kept in one method.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut WeightRef> {
        let i = *self.index.get(name)?;
        Some(&mut self.entries[i].1)
    }

    pub fn all(&self) -> impl Iterator<Item = (&str, &WeightRef)> {
        self.entries.iter().map(|(n, w)| (n.as_str(), w))
    }

    pub fn report(&self) -> LoadReport {
        self.report
    }

    fn insert(&mut self, name: String, w: WeightRef) {
        match self.index.get(&name) {
            Some(&i) => self.entries[i].1 = w,
            None => {
                self.index.insert(name.clone(), self.entries.len());
                self.entries.push((name, w));
            }
        }
    }

    /// The arena size the index asks for, readable WITHOUT loading anything.
    /// With `skip`, the size of the compacted arena that holds every tensor
    /// EXCEPT the named ones.
    pub fn pool_bytes(
        pack_dir: &Path,
        skip: Option<&std::collections::BTreeSet<String>>,
    ) -> Result<u64, String> {
        let path = pack_dir.join("index.txt");
        let lines = read_lines(&path).map_err(|_| format!("cannot open {}", path.display()))?;
        let mut pool = 0u64;
        let mut compact = 0u64;
        let mut align = 0i32;
        for line in &lines {
            if line.starts_with('#') {
                if let Some((a, p)) = parse_header(line) {
                    align = a;
                    pool = p;
                }
                continue;
            }
            if skip.is_none() {
                continue;
            }
            let (fields, n) = split_row(line);
            if n < 7 {
                continue; // sscanf("%255s %d %d %llu %llu %llu %llu") != 7
            }
            let name = fields[0];
            if skip.is_some_and(|s| s.contains(name)) {
                continue;
            }
            let dst_bytes = num(&fields, 6).unwrap_or(0);
            let a = if align > 0 { align as u64 } else { 256 };
            compact += dst_bytes.div_ceil(a) * a;
        }
        if pool == 0 {
            return Err(format!(
                "no '# align ... pool ...' header in {}",
                path.display()
            ));
        }
        Ok(if skip.is_some() { compact } else { pool })
    }

    /// The `code_bits` field of one row, readable WITHOUT loading anything
    /// (0 = the pack stores the tensor unquantized; -1 = no such row).
    pub fn index_code_bits(pack_dir: &Path, name: &str) -> Result<i32, String> {
        let path = pack_dir.join("index.txt");
        let data = std::fs::read(&path).map_err(|_| format!("cannot open {}", path.display()))?;
        for line in String::from_utf8_lossy(&data).lines() {
            if line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split_ascii_whitespace().collect();
            if fields.len() < 10 || fields.iter().any(|f| f.len() > 255) {
                continue; // scanf("%255s %d %d %llu %llu %llu %llu %lld %lld %d") != 10
            }
            let ten_ok = fields[..9].iter().enumerate().all(|(i, f)| {
                // the first nine fields before code_bits must all scan as numbers
                if i == 0 {
                    return true;
                }
                f.parse::<i128>().is_ok()
            });
            if !ten_ok {
                continue;
            }
            if let Ok(bits) = fields[9].parse::<i32>() {
                if name == fields[0] {
                    return Ok(bits);
                }
            }
        }
        Ok(-1)
    }

    /// Load every tensor in `<pack_dir>/index.txt` into `up`'s arena, which must
    /// be at least `pool_bytes` large — the loader checks rather than trusting
    /// the caller, because the failure mode otherwise is a device write past the
    /// end of the arena.
    pub fn load(
        &mut self,
        pack_dir: &Path,
        arena_bytes: u64,
        up: &mut dyn Upload,
        skip: Option<&std::collections::BTreeSet<String>>,
    ) -> Result<(), String> {
        let path = pack_dir.join("index.txt");
        let lines = read_lines(&path)
            .map_err(|_| format!("cannot open {} (run tools/pack_index.py)", path.display()))?;

        let mut pool = 0u64;
        let mut align = 0i32;
        let mut rows: Vec<IndexRow> = Vec::new();
        for line in &lines {
            if line.starts_with('#') {
                if let Some((a, p)) = parse_header(line) {
                    align = a;
                    pool = p;
                }
                continue;
            }
            let (fields, n) = split_row(line);
            if n != 19 || fields.len() < 19 {
                return Err(format!(
                    "index.txt: could not parse a row ({n} of 19 fields)"
                ));
            }
            let g = |i: usize| num(&fields, i);
            let sg = |i: usize| snum(&fields, i);
            let mut kind_i = sg(2)? as i32;
            let mut raw16 = false;
            // index kinds 4 (BF16) and 5 (F16) are copied and take the engine
            // forms of kinds 1 and 3
            if kind_i == 4 || kind_i == 5 {
                raw16 = true;
                kind_i = if kind_i == 4 {
                    WeightKind::Bf16InF32 as i32
                } else {
                    WeightKind::F16InF32 as i32
                };
            }
            rows.push(IndexRow {
                name: fields[0].to_string(),
                file: sg(1)? as i32,
                kind: match kind_i {
                    1 => WeightKind::Bf16InF32,
                    2 => WeightKind::F32,
                    3 => WeightKind::F16InF32,
                    _ => WeightKind::Verbatim,
                },
                raw16,
                src_off: g(3)?,
                src_bytes: g(4)?,
                dst_off: g(5)?,
                dst_bytes: g(6)?,
                ne0: sg(7)?,
                ne1: sg(8)?,
                code_bits: sg(9)? as i32,
                code_bias: sg(10)? as i32,
                group_elems: sg(11)? as i32,
                codebook: sg(12)? as i32,
                has_offset: sg(13)? as i32,
                codes_bytes: g(14)?,
                scales_bytes: g(15)?,
                offset_bytes: g(16)?,
                scales_fp16: sg(17)? != 0,
                act_kind: sg(18)? as i32,
            });
        }

        if pool == 0 || rows.is_empty() {
            return Err("index.txt has no header or no rows".into());
        }
        // A skip set compacts the arena: kept rows are re-placed in index order
        // at the index's alignment; skipped rows keep their metadata and get no
        // bytes.
        let mut skipped = vec![false; rows.len()];
        if let Some(skip) = skip {
            let a = if align > 0 { align as u64 } else { 256 };
            let mut at = 0u64;
            for (i, row) in rows.iter_mut().enumerate() {
                if skip.contains(&row.name) {
                    skipped[i] = true;
                    continue;
                }
                row.dst_off = at;
                at += row.dst_bytes.div_ceil(a) * a;
            }
            pool = at;
        }
        if arena_bytes < pool {
            return Err(format!(
                "arena is {arena_bytes} B but the index needs {pool}"
            ));
        }

        self.entries.clear();
        self.index.clear();
        self.report = LoadReport::default();

        let mut cur: Option<(i32, File)> = None;
        for (i, r) in rows.iter().enumerate() {
            if skipped[i] {
                let mut wr = ref_from_row(r);
                wr.arena_off = None;
                wr.resident = false;
                self.insert(r.name.clone(), wr);
                self.report.tensors += 1;
                continue;
            }
            if r.code_bits != 0 && r.dst_bytes == 0 {
                // a native pack's row that carries a shape only — the GGUF form must serve it
                return Err(r.name.clone()
                    + ": this pack holds the tensor only in its GGUF form (run with --native SHARD1)");
            }
            if cur.as_ref().is_none_or(|(id, _)| *id != r.file) {
                let fn_name = pack_file_name(r.file).unwrap_or("?");
                let p = pack_dir.join(fn_name);
                let f = File::open(&p).map_err(|_| format!("cannot open {}", p.display()))?;
                cur = Some((r.file, f));
            }
            let file = &mut cur.as_mut().unwrap().1;

            // ---- the segment list: this tensor's bytes, plane by plane
            #[derive(Clone, Copy, PartialEq)]
            enum Conv {
                Copy,
                Bf16High,
                ToF16,
                WidenF16,
            }
            #[derive(Clone, Copy)]
            struct Seg {
                src_off: u64,
                src_bytes: u64,
                dst_off: u64,
                dst_bytes: u64,
                conv: Conv,
            }
            let mut segs: Vec<Seg> = Vec::new();
            let mut bad: Option<String> = None;
            if r.codes_bytes != 0 {
                // A quantized tensor: codes, then scales, then the optional
                // offset plane, each contiguous in the source span in that order.
                segs.push(Seg {
                    src_off: 0,
                    src_bytes: r.codes_bytes,
                    dst_off: 0,
                    dst_bytes: r.codes_bytes,
                    conv: Conv::Copy,
                });
                if r.scales_bytes != 0 {
                    let src_scale_bytes = if r.scales_fp16 {
                        r.scales_bytes / 2
                    } else {
                        r.scales_bytes
                    };
                    if r.scales_fp16
                        && (r.scales_bytes % 2 != 0 || src_scale_bytes * 2 != r.scales_bytes)
                    {
                        bad = Some(r.name.clone() + ": an fp16 scale plane of an odd byte count");
                    }
                    segs.push(Seg {
                        src_off: r.codes_bytes,
                        src_bytes: src_scale_bytes,
                        dst_off: r.codes_bytes,
                        dst_bytes: r.scales_bytes,
                        conv: if r.scales_fp16 {
                            Conv::WidenF16
                        } else {
                            Conv::Copy
                        },
                    });
                }
                if r.offset_bytes != 0 {
                    let after = r.codes_bytes
                        + if r.scales_fp16 {
                            r.scales_bytes / 2
                        } else {
                            r.scales_bytes
                        };
                    segs.push(Seg {
                        src_off: after,
                        src_bytes: r.offset_bytes,
                        dst_off: r.codes_bytes + r.scales_bytes,
                        dst_bytes: r.offset_bytes,
                        conv: Conv::Copy,
                    });
                }
            } else {
                // Everything else is one run whose transform is the row's kind.
                let conv = if r.raw16 {
                    Conv::Copy
                } else if r.kind == WeightKind::Bf16InF32 {
                    Conv::Bf16High
                } else if r.kind == WeightKind::F16InF32 {
                    Conv::ToF16
                } else {
                    Conv::Copy
                };
                let elems = (r.ne0 * if r.ne1 > 0 { r.ne1 } else { 1 }) as u64;
                if conv != Conv::Copy && (r.src_bytes != elems * 4 || r.dst_bytes != elems * 2) {
                    bad = Some(format!(
                        "{}: a promoted tensor of {elems} elements is {} B in and {} B out, not {}/{}",
                        r.name, r.src_bytes, r.dst_bytes,
                        elems * 4,
                        elems * 2
                    ));
                }
                segs.push(Seg {
                    src_off: 0,
                    src_bytes: r.src_bytes,
                    dst_off: 0,
                    dst_bytes: r.dst_bytes,
                    conv,
                });
            }
            let sum_src: u64 = segs.iter().map(|s| s.src_bytes).sum();
            let sum_dst: u64 = segs.iter().map(|s| s.dst_bytes).sum();
            // The segment list is the ONLY thing standing between a mis-sized
            // plane and a wrong byte offset, so it is checked against both
            // recorded totals, not trusted.
            if bad.is_none() && (sum_src != r.src_bytes || sum_dst != r.dst_bytes) {
                bad = Some(format!(
                    "{}: the segments cover {sum_src}/{sum_dst} B but the index says {}/{r_dst} - the plane layout this loader built is not the one the index describes",
                    r.name, r.src_bytes, r_dst = r.dst_bytes
                ));
            }
            if let Some(msg) = bad {
                return Err(msg);
            }

            for s in &segs {
                let widening = matches!(s.conv, Conv::WidenF16);
                if widening && (s.src_bytes & 1 == 1) {
                    return Err(r.name.clone() + ": an fp16 plane with an odd source byte count");
                }
                let mut done = 0u64;
                let mut stage_in: Vec<u8> = Vec::new();
                while done < s.src_bytes {
                    let mut n = CHUNK.min(s.src_bytes - done);
                    // WIDENING NEVER SPLITS AN ELEMENT: a full CHUNK cannot, but
                    // a plane tail of odd length would — half an fp16 is a
                    // plausible-looking scale.
                    if widening && (n & 1 == 1) {
                        if n == 1 {
                            return Err(r.name.clone() + ": an fp16 plane ending on a half element");
                        }
                        n -= 1;
                    }
                    stage_in.resize(n as usize, 0);
                    read_at(file, r.src_off + s.src_off + done, &mut stage_in, &r.name)?;
                    let mut staged: Vec<u8> = Vec::new();
                    let (host_src, out_at): (&[u8], u64) = match s.conv {
                        Conv::Copy => (&stage_in, s.dst_off + done),
                        Conv::Bf16High | Conv::ToF16 => {
                            // THE HIGH 16 BITS ARE THE BF16 ENCODING, exactly;
                            // ToF16 is a real round-to-nearest-even conversion.
                            let take_high = matches!(s.conv, Conv::Bf16High);
                            for chunk in stage_in.as_chunks::<4>().0 {
                                let v = u32::from_le_bytes(*chunk);
                                let h = if take_high {
                                    (v >> 16) as u16
                                } else {
                                    f16_from_f32(f32::from_bits(v))
                                };
                                staged.extend_from_slice(&h.to_le_bytes());
                            }
                            (&staged, s.dst_off + done / 2)
                        }
                        Conv::WidenF16 => {
                            // fp16 bits -> f32, EXACT: this is why the engine may
                            // widen 90 scale planes at all.
                            for chunk in stage_in.as_chunks::<2>().0 {
                                let h = u16::from_le_bytes(*chunk);
                                staged.extend_from_slice(&f32_from_f16(h).to_bits().to_le_bytes());
                            }
                            (&staged, s.dst_off + done * 2)
                        }
                    };
                    if up.upload(r.dst_off + out_at, host_src).is_err() {
                        return Err(format!("cudaMemcpy failed for {}", r.name));
                    }
                    done += n;
                }
            }

            let mut wr = ref_from_row(r);
            wr.arena_off = Some(r.dst_off);
            self.insert(r.name.clone(), wr);
            self.report.tensors += 1;
            if (r.kind == WeightKind::Bf16InF32 || r.kind == WeightKind::F16InF32) && !r.raw16 {
                self.report.re_rounded += 1;
                self.report.bytes_saved += r.src_bytes - r.dst_bytes;
            }
        }

        self.report.arena_bytes = pool;
        Ok(())
    }
}

impl Default for WeightTable {
    fn default() -> Self {
        WeightTable::new()
    }
}

fn ref_from_row(r: &IndexRow) -> WeightRef {
    WeightRef {
        arena_off: None, // set by the caller once the bytes are where they belong
        bytes: r.dst_bytes,
        ne0: r.ne0,
        ne1: r.ne1,
        elements: r.ne0 * if r.ne1 > 0 { r.ne1 } else { 1 },
        kind: r.kind,
        code_bits: r.code_bits,
        code_bias: r.code_bias,
        group_elems: r.group_elems,
        codebook_iq4nl: r.codebook != 0,
        has_offset: r.has_offset != 0,
        codes_bytes: r.codes_bytes,
        scales_bytes: r.scales_bytes,
        offset_bytes: r.offset_bytes,
        src_off: r.src_off,
        src_bytes: r.src_bytes,
        file_id: r.file,
        scales_fp16: r.scales_fp16,
        act_kind: r.act_kind,
        native_off: None,
        native_q8_1: None,
        native_type: -1,
        resident: true,
    }
}

fn read_at(f: &mut File, off: u64, dst: &mut [u8], what: &str) -> Result<(), String> {
    f.seek(SeekFrom::Start(off))
        .map_err(|_| format!("{what}: seek failed"))?;
    f.read_exact(dst)
        .map_err(|_| format!("{what}: short read"))?;
    Ok(())
}

/// Path helper kept separate so a caller can name a pack directory without the
/// loader's internals.
pub fn index_path(pack_dir: &Path) -> PathBuf {
    pack_dir.join("index.txt")
}
