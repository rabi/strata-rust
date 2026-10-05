//! Byte-parity replay of the native weight path: `src/core/weights.cpp`,
//! `src/core/native_dense.cpp` and the byte tables of `native_mmvq.hpp`.
//!
//! `tools/native_dense_corpus.cpp` compiles the REAL C++ (CUDA calls `--wrap`'ped
//! to host memory, block sizes taken from `ggml-common.h`) and prints one
//! `CORPUS|…` line per observation: index parse verdicts, per-plane upload
//! plans, fnv1a64 of the exact bytes handed to `cudaMemcpy`, fnv1a64 of the
//! arena after every load, every error string, `served_names` sets, and
//! `WeightRef` fields. That output is checked in as `golden/native_dense.txt`;
//! the fixture files the C++ wrote are embedded in it as hex, so this test
//! rebuilds the same pack directories and GGUF shards and replays every scenario
//! through the Rust port. The two line streams must be identical.
//!
//! Regenerate (needs a Strata checkout; build line in the harness header):
//!   ./target/corpus/ndcorpus | <normalize the temp path to <TMP>> \
//!       > crates/strata-core/tests/golden/native_dense.txt
//!
//! One call-shape difference from the C++, noted because it moves an error
//! message: `NativeDense::load` takes parsed shards rather than paths, so the
//! driver opens each shard and maps a GGUF parse failure onto the same
//! `native dense: …` text the C++'s `catch` produces around `GgufFile gguf(path)`
//! inside its own loop.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use strata_artifact::gguf::{check_architecture, GgufFile, Qwen4ExpGuard, TensorInfo};
use strata_core::fnv;
use strata_core::native_dense::{
    keep_unquantized_ple_key, served_names, Arch, LayerRange, NativeCandidate, NativeDense,
    NativeShard, NativeUpload,
};
use strata_core::native_mm::{
    native_mmvq_supported, native_mmvq_weight_bytes, native_q8_1_bytes, weight_block,
};
use strata_core::weights::{Upload, WeightRef, WeightTable};

const GOLDEN: &str = include_str!("golden/native_dense.txt");

const QKV: &str = "blk.0.attn_qkv.weight";
const KEY: &str = "blk.1.ple_key.weight";
const TMP: &str = "<TMP>";

// ------------------------------------------------------------------ golden I/O

/// Every observation line in golden order; the fixture dump is input, not output.
fn expected() -> Vec<String> {
    GOLDEN
        .lines()
        .filter(|l| l.starts_with("CORPUS|") && !is_dump(l))
        .map(str::to_string)
        .collect()
}

fn is_dump(line: &str) -> bool {
    line.starts_with("CORPUS|BLOB|") || line.starts_with("CORPUS|FILE|")
}

fn fixtures() -> BTreeMap<String, Vec<u8>> {
    let mut blobs: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    let mut pending: Vec<(String, u32)> = Vec::new();
    for line in GOLDEN.lines() {
        if let Some(rest) = line.strip_prefix("CORPUS|BLOB|") {
            let (id, hex) = rest.split_once('|').expect("BLOB line");
            blobs.insert(id.parse().unwrap(), unhex(hex));
        } else if let Some(rest) = line.strip_prefix("CORPUS|FILE|") {
            let (rel, id) = rest.split_once('|').expect("FILE line");
            pending.push((rel.to_string(), id.parse().unwrap()));
        }
    }
    pending
        .into_iter()
        .map(|(rel, id)| (rel, blobs[&id].clone()))
        .collect()
}

fn unhex(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The C++ harness's temp directory, rebuilt from the golden and written to disk
/// because `WeightTable` reads a pack from the filesystem.
struct Tree {
    root: PathBuf,
}

impl Tree {
    fn new(files: &BTreeMap<String, Vec<u8>>) -> Tree {
        let root = std::env::temp_dir().join(format!(
            "strata-nd-replay-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&root).expect("temp dir");
        for (rel, bytes) in files {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).expect("fixture dir");
            std::fs::write(&p, bytes).expect("fixture write");
        }
        Tree { root }
    }
    fn path(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }
    /// the golden prints its own temp root as `<TMP>`; so does this driver
    fn norm(&self, line: &str) -> String {
        line.replace(&self.root.display().to_string(), TMP)
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

// ------------------------------------------------------------------ fake device
//
// The C++ corpus `--wrap`s cudaMalloc/cudaMemcpy onto one host slab: allocations
// bump a 256-aligned cursor, a fresh allocation is poisoned with 0xa5 so an
// unread destination stays visible, and every copy is recorded with an fnv of its
// SOURCE bytes and a name for where the destination lives (`arena:<pack>:<off>`
// for a registered arena, `outside` for anything else — which is what the native
// projections are, because only the arenas are registered). The replay does the
// same, so a `CORPUS|WRITE` line compares the whole load plan.

const SLAB: usize = 32 << 20;

struct Dev {
    slab: Vec<u8>,
    used: usize,
    regions: Vec<(usize, usize, String)>,
    writes: Vec<(String, u64, u64)>,
    recording: bool,
    /// the arena the next `WeightTable::load` writes into, as a slab offset
    arena_base: usize,
}

impl Dev {
    fn new() -> Dev {
        Dev {
            slab: vec![0; SLAB],
            used: 0,
            regions: Vec::new(),
            writes: Vec::new(),
            recording: false,
            arena_base: 0,
        }
    }
    fn malloc(&mut self, n: usize) -> usize {
        assert!(self.used + n <= SLAB, "slab exhausted");
        let at = self.used;
        self.slab[at..at + n].fill(0xa5);
        self.used = (self.used + n + 255) & !255;
        at
    }
    fn reg(&mut self, at: usize, n: usize, name: String) {
        self.regions.push((at, n, name));
    }
    fn where_(&self, at: usize, n: usize) -> String {
        for (base, len, name) in &self.regions {
            if at >= *base && at < base + len && at + n <= base + len {
                return format!("{name}:{}", at - base);
            }
        }
        "outside".to_string()
    }
    fn copy(&mut self, at: usize, src: &[u8]) {
        if self.recording {
            let src_fnv = fnv(src);
            let dst = self.where_(at, src.len());
            self.writes.push((dst, src.len() as u64, src_fnv));
        }
        self.slab[at..at + src.len()].copy_from_slice(src);
    }
    fn fnv_at(&self, at: usize, len: usize) -> u64 {
        fnv(&self.slab[at..at + len])
    }
    fn hex_at(&self, at: usize, len: usize) -> String {
        hex(&self.slab[at..at + len])
    }
    fn clear(&mut self) {
        self.writes.clear();
    }
    fn print_writes(&self, tag: &str, out: &mut Vec<String>) {
        for (dst, bytes, src_fnv) in &self.writes {
            out.push(format!(
                "CORPUS|WRITE|{tag}|{dst}|bytes={bytes}|src_fnv={src_fnv}"
            ));
        }
    }
}

impl Upload for Dev {
    fn upload(&mut self, arena_off: u64, bytes: &[u8]) -> Result<(), String> {
        self.copy(self.arena_base + arena_off as usize, bytes);
        Ok(())
    }
}

impl NativeUpload for Dev {
    fn alloc(&mut self, bytes: u64) -> Result<u64, String> {
        Ok(self.malloc(bytes as usize) as u64)
    }
    fn upload(&mut self, at: u64, bytes: &[u8]) -> Result<(), String> {
        self.copy(at as usize, bytes);
        Ok(())
    }
    fn free(&mut self, _at: u64, _bytes: u64) {}
}

// ------------------------------------------------------------------ shards

/// Open every shard, in order, the way the C++ opens them inside its own loop.
fn open_shards(tree: &Tree, rels: &[&str]) -> Result<Vec<GgufFile>, String> {
    let mut out = Vec::new();
    for r in rels {
        let g = GgufFile::open_memory(tree.path(r))
            .map_err(|e| format!("native dense: {}", tree.norm(&e)))?;
        out.push(g);
    }
    Ok(out)
}

fn shards_of(gs: &[GgufFile]) -> Vec<NativeShard<'_>> {
    gs.iter().map(shard_of).collect()
}

fn shard_of(g: &GgufFile) -> NativeShard<'_> {
    let num = |k: &str| g.get(k).map(|v| v.u);
    let arch = match g.get("general.architecture") {
        None => Arch::Absent,
        Some(_) => {
            let why = check_architecture(g, &Qwen4ExpGuard::default());
            if why.is_empty() {
                Arch::Validated
            } else {
                Arch::Failed(why)
            }
        }
    };
    NativeShard {
        path: g.path().display().to_string(),
        tensors: g.tensors().iter().map(|t| candidate(g, t)).collect(),
        data_start: g.data_start(),
        file_size: g.file_size(),
        arch,
        split_count: num("split.count"),
        split_no: num("split.no"),
        split_tensors: num("split.tensors.count"),
    }
}

fn candidate<'a>(g: &'a GgufFile, t: &TensorInfo) -> NativeCandidate<'a> {
    NativeCandidate {
        name: t.name.clone(),
        ggml_type: t.dtype as i32,
        shape: t.shape.clone(),
        offset: t.offset,
        // a shard whose header lies about its payload never gets this far in a
        // load that succeeds, so an empty slice is the honest answer here
        data: g.tensor_bytes(t).unwrap_or(&[]),
    }
}

// ------------------------------------------------------------------ lines

fn ref_line(tag: &str, name: &str, w: Option<&WeightRef>) -> String {
    let Some(w) = w else {
        return format!("CORPUS|REF|{tag}|{name}|absent");
    };
    format!(
        "CORPUS|REF|{tag}|{name}|bytes={bytes}|ne={ne0},{ne1}|elements={elements}|kind={kind}\
         |quantized={q}|resident={res}|native={nat}|native_type={nt}|native_q8_1={nq}|codes={cd}\
         |scales={sc}|offset={of}|fp16={fp16}|act={act}|src={file}:{so}+{sb}|code_bits={cb}\
         |group={ge}|bias={bi}|iq4={iq}",
        bytes = w.bytes,
        ne0 = w.ne0,
        ne1 = w.ne1,
        elements = w.elements,
        kind = w.kind as i32,
        q = w.quantized() as i32,
        res = w.resident as i32,
        nat = w.native_off.is_some() as i32,
        nt = w.native_type,
        nq = w.native_q8_1.is_some() as i32,
        cd = w.codes_bytes,
        sc = w.scales_bytes,
        of = w.offset_bytes,
        fp16 = w.scales_fp16 as i32,
        act = w.act_kind,
        file = w.file_id,
        so = w.src_off,
        sb = w.src_bytes,
        cb = w.code_bits,
        ge = w.group_elems,
        bi = w.code_bias,
        iq = w.codebook_iq4nl as i32,
    )
}

fn join(set: &BTreeSet<String>) -> String {
    let mut out = String::new();
    for n in set {
        out.push_str(n);
        out.push(',');
    }
    out
}

/// the message the C++ prints in its `err=` field: empty on success, the
/// reader's or loader's own text on failure.
fn err_of<T>(tree: &Tree, r: &Result<T, String>) -> String {
    r.as_ref().err().map_or_else(String::new, |e| tree.norm(e))
}

// ------------------------------------------------------------------ scenario 1:
// byte-format tables, every supported type and every refusal message.
fn scenario_bytes(out: &mut Vec<String>) {
    out.push(format!(
        "CORPUS|Q81SIZE|bytes={}",
        native_q8_1_bytes(32, 1).unwrap()
    ));
    for t in [
        2, 6, 7, 8, 20, 42, 11, 12, 13, 14, 23, 16, 17, 18, 21, 22, 29, 0, 30, 255,
    ] {
        let (elems, bytes) = match weight_block(t) {
            Some(g) => (g.0 as i64, g.1 as i64),
            None => (0, -1),
        };
        out.push(format!(
            "CORPUS|BLOCKSIZE|t={t}|elems={elems}|bytes={bytes}"
        ));
    }
    for (t, n_in, n_out, show) in table_cases() {
        let mut line = format!("CORPUS|BYTES|t={t}|n_in={n_in}");
        if show {
            line.push_str(&format!("|n_out={n_out}"));
        }
        line.push_str(&format!("|supported={}", native_mmvq_supported(t) as i32));
        match native_mmvq_weight_bytes(t, n_in, n_out) {
            Ok(b) => line.push_str(&format!("|bytes={b}")),
            Err(e) => line.push_str(&format!("|throw={e}")),
        }
        out.push(line);
    }
    for (n_in, ncols) in [
        (256, 1),
        (2560, 8),
        (32, 1),
        (255, 1),
        (256, 0),
        (256, 9),
        (2147483647, 2147483647),
    ] {
        let mut line = format!("CORPUS|Q81|n_in={n_in}|ncols={ncols}");
        match native_q8_1_bytes(n_in, ncols) {
            Ok(b) => line.push_str(&format!("|bytes={b}")),
            Err(e) => line.push_str(&format!("|throw={e}")),
        }
        out.push(line);
    }
}

/// The `(type, n_in, n_out, print n_out)` cases, in the harness's order: every
/// supported type across the sizes a projection actually takes, then the types
/// the table does not know, then the shapes the shape check refuses.
fn table_cases() -> Vec<(i32, i32, i32, bool)> {
    let mut v = Vec::new();
    for t in [
        2, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 42,
    ] {
        for n_in in [32, 64, 256, 512, 1024, 2560] {
            v.push((t, n_in, 4, false));
        }
    }
    for t in [
        0, 1, 3, 4, 5, 9, 10, 15, 19, 24, 25, 26, 27, 28, 30, 31, 41, 43, -1, 1000,
    ] {
        v.push((t, 256, 4, false));
    }
    for c in [
        (2, 250, 4),
        (42, 128, 0),
        (8, 0, 1),
        (14, 2147483621, 2147483647),
        (2, -32, 4),
        (42, 64, -1),
    ] {
        v.push((c.0, c.1, c.2, true));
    }
    v
}

// ------------------------------------------------------------------ scenario 2:
// the C++ test's fixture, in generate.cpp's order: served_names ->
// keep_unquantized_ple_key -> pool_bytes -> WeightTable::load -> NativeDense::load.
fn scenario_fixture(tree: &Tree, dev: &mut Dev, out: &mut Vec<String>, tag: &str) {
    let pack = tree.path(tag);
    let shard = format!("{tag}.gguf");
    let gs = open_shards(tree, &[&shard]).unwrap();
    let shards = shards_of(&gs);

    let mut skip = served_names(&shards, true);
    out.push(format!(
        "CORPUS|SERVED|{tag}|ok=1|err=|names={}",
        join(&skip)
    ));

    let kept = keep_unquantized_ple_key(&pack, &mut skip);
    out.push(format!(
        "CORPUS|KEEP|{tag}|ok={}|err={}|key_skipped={}|qkv_skipped={}",
        kept.is_ok() as i32,
        kept.as_ref()
            .err()
            .map_or_else(String::new, |e| tree.norm(e)),
        skip.contains(KEY) as i32,
        skip.contains(QKV) as i32,
    ));

    let pool = WeightTable::pool_bytes(&pack, Some(&skip)).unwrap_or(0);
    out.push(format!("CORPUS|POOL|{tag}|ok=1|err=|pool={pool}"));

    let cap = pool;
    let cap = if cap > 0 { cap } else { 256 };
    let base = dev.malloc(cap as usize);
    dev.reg(base, cap as usize, format!("arena:{tag}"));
    dev.arena_base = base;
    let mut wt = WeightTable::new();
    dev.clear();
    dev.recording = true;
    let ok = wt.load(&pack, cap, &mut *dev, Some(&skip));
    dev.recording = false;
    out.push(format!(
        "CORPUS|LOAD|{tag}|ok={}|err={}|writes={}",
        ok.is_ok() as i32,
        err_of(tree, &ok),
        dev.writes.len(),
    ));
    dev.print_writes(tag, out);
    let r = wt.report();
    out.push(format!(
        "CORPUS|REPORT|{tag}|tensors={}|arena={}|re_rounded={}|bytes_saved={}",
        r.tensors, r.arena_bytes, r.re_rounded, r.bytes_saved
    ));
    out.push(ref_line(tag, QKV, wt.find(QKV)));
    out.push(ref_line(tag, KEY, wt.find(KEY)));
    out.push(format!(
        "CORPUS|ARENA|{tag}|fnv={}",
        dev.fnv_at(base, cap as usize)
    ));

    let mut dense = NativeDense::default();
    dev.clear();
    dev.recording = true;
    let ok = dense.load(&shards, &mut wt, &mut *dev, true, (0, -1), LayerRange::ALL);
    dev.recording = false;
    let upload_bytes: u64 = dev.writes.iter().map(|w| w.1).sum();
    out.push(format!(
        "CORPUS|DENSE|{tag}|ok={}|err={}|tensors={}|weight_bytes={}|uploads={}|upload_bytes={upload_bytes}",
        ok.is_ok() as i32,
        err_of(tree, &ok),
        dense.tensor_count(),
        dense.weight_bytes(),
        dev.writes.len(),
    ));
    dev.print_writes(tag, out);
    out.push(ref_line(tag, QKV, wt.find(QKV)));
    out.push(ref_line(tag, KEY, wt.find(KEY)));
    out.push(format!(
        "CORPUS|ARENA|{tag}|fnv={}",
        dev.fnv_at(base, cap as usize)
    ));
}

// ------------------------------------------------------------------ scenario 3:
// plan v0.3 P6's tensor kinds through the real loader — a quantized plane split,
// an fp16 scale plane widened, a BF16 promotion shrunk, an f16 promotion rounded,
// raw16 kinds 4/5 copied — plus every refusal the loader can print.
fn scenario_kinds(tree: &Tree, dev: &mut Dev, out: &mut Vec<String>) {
    let base = dev.malloc(65536);
    dev.reg(base, 65536, "arena:kinds".to_string());
    dev.arena_base = base;
    let mut wt = WeightTable::new();
    dev.clear();
    dev.recording = true;
    let ok = wt.load(&tree.path("kinds"), 65536, &mut *dev, None);
    dev.recording = false;
    out.push(format!(
        "CORPUS|LOAD|kinds|ok={}|err={}|writes={}",
        ok.is_ok() as i32,
        err_of(tree, &ok),
        dev.writes.len(),
    ));
    dev.print_writes("kinds", out);
    let r = wt.report();
    out.push(format!(
        "CORPUS|REPORT|kinds|tensors={}|arena={}|re_rounded={}|bytes_saved={}",
        r.tensors, r.arena_bytes, r.re_rounded, r.bytes_saved
    ));
    for n in ["t.q", "t.bf", "t.h", "t.f32", "t.raw4", "t.raw5", "t.codes"] {
        out.push(ref_line("kinds", n, wt.find(n)));
    }
    out.push(format!(
        "CORPUS|ARENA|kinds|q={}|bf={}|h={}|f32={}|r4={}|r5={}|codes={}",
        dev.fnv_at(base, 128),
        dev.fnv_at(base + 128, 24),
        dev.fnv_at(base + 152, 24),
        dev.fnv_at(base + 176, 32),
        dev.fnv_at(base + 208, 64),
        dev.fnv_at(base + 272, 64),
        dev.fnv_at(base + 336, 64),
    ));
    out.push(format!(
        "CORPUS|HEX|kinds|bf={}|h={}",
        dev.hex_at(base + 128, 24),
        dev.hex_at(base + 152, 24)
    ));

    // one bad index per case: the exact message the loader prints
    for case in [
        "fields",
        "header_fields",
        "shape_only",
        "no_header",
        "odd_fp16_plane",
        "promote_mismatch",
        "seg_mismatch",
        "bad_file",
        "good",
    ] {
        let dir = tree.path(&format!("bad-{case}"));
        let pool = WeightTable::pool_bytes(&dir, None);
        out.push(format!(
            "CORPUS|POOL|bad-{case}|ok={}|err={}|pool={}",
            pool.is_ok() as i32,
            err_of(tree, &pool),
            pool.unwrap_or(0)
        ));
        let base = dev.malloc(65536);
        dev.reg(base, 65536, format!("arena:bad-{case}"));
        dev.arena_base = base;
        let mut wt2 = WeightTable::new();
        dev.clear();
        dev.recording = true;
        let ok = wt2.load(&dir, 65536, &mut *dev, None);
        dev.recording = false;
        out.push(format!(
            "CORPUS|LOAD|bad-{case}|ok={}|err={}|writes={}|tensors={}",
            ok.is_ok() as i32,
            err_of(tree, &ok),
            dev.writes.len(),
            wt2.report().tensors,
        ));
        let bits = WeightTable::index_code_bits(&dir, "t.q");
        out.push(format!(
            "CORPUS|BITS|bad-{case}|ok={}|err={}|bits={}",
            bits.is_ok() as i32,
            err_of(tree, &bits),
            bits.unwrap_or(99),
        ));
    }

    // the arena-too-small message, with an otherwise good index
    {
        let base = dev.malloc(65536);
        dev.reg(base, 65536, "arena:arena_small".to_string());
        dev.arena_base = base;
        let mut wt3 = WeightTable::new();
        dev.clear();
        dev.recording = true;
        let ok = wt3.load(&tree.path("bad-arena_small"), 4, &mut *dev, None);
        dev.recording = false;
        out.push(format!(
            "CORPUS|LOAD|bad-arena_small|ok={}|err={}|writes={}",
            ok.is_ok() as i32,
            err_of(tree, &ok),
            dev.writes.len(),
        ));
    }

    // index_code_bits on a missing directory, and a missing tensor row. The C++
    // prints the same `err` and `bits` variables for both, so a failure leaves
    // its message behind for the next line; the golden records that.
    let mut err = String::new();
    let mut bits = 99;
    match WeightTable::index_code_bits(&tree.path("nowhere"), KEY) {
        Ok(b) => bits = b,
        Err(e) => err = tree.norm(&e),
    }
    out.push(format!("CORPUS|BITS|no_index|ok=0|err={err}|bits={bits}"));
    match WeightTable::index_code_bits(&tree.path("kinds"), "t.absent") {
        Ok(b) => bits = b,
        Err(e) => err = tree.norm(&e),
    }
    out.push(format!("CORPUS|BITS|no_row|ok=1|err={err}|bits={bits}"));

    // keep_unquantized_ple_key when the key is not in skip (the index is never opened)
    let mut skip_nokey = BTreeSet::from([QKV.to_string()]);
    let kerr = keep_unquantized_ple_key(&tree.path("nowhere"), &mut skip_nokey);
    out.push(format!(
        "CORPUS|KEEP|no_key_in_skip|err={}|skipped={}",
        kerr.as_ref()
            .err()
            .map_or_else(String::new, |e| tree.norm(e)),
        skip_nokey.len(),
    ));
}

// ------------------------------------------------------------------ scenario 4:
// the split arbitration, the stage range, and the two branches after the
// duplicate check.
fn scenario_shards(tree: &Tree, dev: &mut Dev, out: &mut Vec<String>) {
    struct Case {
        shards: &'static [&'static str],
        ple: bool,
        lo: i64,
        hi: i64,
        why: &'static str,
    }
    let cases = [
        Case {
            shards: &["sh0.gguf"],
            ple: true,
            lo: 0,
            hi: -1,
            why: "single",
        },
        Case {
            shards: &["sh0.gguf", "sh1.gguf"],
            ple: true,
            lo: 0,
            hi: -1,
            why: "continuation",
        },
        Case {
            shards: &["sh0.gguf", "sh2.gguf"],
            ple: true,
            lo: 0,
            hi: -1,
            why: "bad_continuation",
        },
        Case {
            shards: &["sh0.gguf", "sh0.gguf"],
            ple: true,
            lo: 0,
            hi: -1,
            why: "duplicate_number",
        },
        Case {
            shards: &["sh1.gguf"],
            ple: true,
            lo: 0,
            hi: -1,
            why: "no_architecture",
        },
        Case {
            shards: &["sh0.gguf"],
            ple: true,
            lo: 1,
            hi: 2,
            why: "stage_holds_layer1",
        },
        Case {
            shards: &["sh0.gguf"],
            ple: false,
            lo: 0,
            hi: -1,
            why: "ple_not_native",
        },
        Case {
            shards: &[],
            ple: true,
            lo: 0,
            hi: -1,
            why: "no_shard",
        },
    ];
    for (i, c) in cases.iter().enumerate() {
        let gs = open_shards(tree, c.shards).unwrap();
        let shards = shards_of(&gs);
        let skip = served_names(&shards, c.ple);
        out.push(format!(
            "CORPUS|SERVED|c{i}_{}|ok=1|err=|names={}",
            c.why,
            join(&skip)
        ));

        let base = dev.malloc(65536);
        dev.reg(base, 65536, format!("arena:c{i}"));
        dev.arena_base = base;
        let mut wt = WeightTable::new();
        let table_err = match wt.load(&tree.path(&format!("c{i}")), 65536, &mut *dev, Some(&skip)) {
            Ok(()) => String::new(),
            Err(e) => tree.norm(&e),
        };
        let table_ok = wt.report().tensors > 0;
        let mut dense = NativeDense::default();
        dev.clear();
        dev.recording = true;
        let ok = dense.load(
            &shards,
            &mut wt,
            &mut *dev,
            c.ple,
            (c.lo, c.hi),
            LayerRange::ALL,
        );
        dev.recording = false;
        out.push(format!(
            "CORPUS|DENSE|c{i}_{}|table_ok={}|table_err={}|ok={}|err={}|tensors={}|weight_bytes={}|uploads={}",
            c.why,
            table_ok as i32,
            table_err,
            ok.is_ok() as i32,
            err_of(tree, &ok),
            dense.tensor_count(),
            dense.weight_bytes(),
            dev.writes.len(),
        ));
        dev.print_writes(c.why, out);
    }

    // the process-wide layer range, read by the NEXT load (the C++ keeps it in a
    // file-static; the Rust port takes it as an argument)
    {
        let gs = open_shards(tree, &["range.gguf"]).unwrap();
        let shards = shards_of(&gs);
        let skip = served_names(&shards, false);
        let base = dev.malloc(65536);
        dev.reg(base, 65536, "arena:range".to_string());
        dev.arena_base = base;
        let mut wt = WeightTable::new();
        let _ = wt.load(&tree.path("range"), 65536, &mut *dev, Some(&skip));
        let mut dense = NativeDense::default();
        dev.clear();
        dev.recording = true;
        let range = LayerRange { lo: 0, hi: 1 };
        let ok = dense.load(&shards, &mut wt, &mut *dev, false, (0, -1), range);
        dev.recording = false;
        out.push(format!(
            "CORPUS|RANGE|ok={}|err={}|tensors={}|uploads={}",
            ok.is_ok() as i32,
            err_of(tree, &ok),
            dense.tensor_count(),
            dev.writes.len(),
        ));
        let again = dense.load(&shards, &mut wt, &mut *dev, false, (0, -1), range);
        out.push(format!(
            "CORPUS|AGAIN|ok={}|err={}",
            again.is_ok() as i32,
            err_of(tree, &again),
        ));
    }

    // an eligible name that is not in the canonical table, and an unsupported
    // type: the two branches after the duplicate check
    {
        let gs = open_shards(tree, &["absent.gguf"]).unwrap();
        let shards = shards_of(&gs);
        let base = dev.malloc(65536);
        dev.reg(base, 65536, "arena:absent".to_string());
        dev.arena_base = base;
        // The C++ shares ONE `err` string across build_table and the load, and
        // `native_dense.cpp` assigns `check_architecture()`'s result to it, so a
        // load that reaches a valid architecture CLEARS the previous message. The
        // Rust port returns Result; the clearing is modelled here as "Ok means the
        // message is empty", which is what the golden shows.
        let mut wt = WeightTable::new();
        let _ = wt.load(&tree.path("absent"), 65536, &mut *dev, None);
        let mut dense = NativeDense::default();
        let ok = dense.load(&shards, &mut wt, &mut *dev, false, (0, -1), LayerRange::ALL);
        out.push(format!("CORPUS|ABSENT|ok=0|err={}", err_of(tree, &ok)));

        let base2 = dev.malloc(65536);
        dev.reg(base2, 65536, "arena:absent2".to_string());
        dev.arena_base = base2;
        let mut wt2 = WeightTable::new();
        let _ = wt2.load(&tree.path("absent2"), 65536, &mut *dev, None);
        let mut dense2 = NativeDense::default();
        dev.clear();
        dev.recording = true;
        let ok2 = dense2.load(
            &shards,
            &mut wt2,
            &mut *dev,
            false,
            (0, -1),
            LayerRange::ALL,
        );
        dev.recording = false;
        out.push(format!(
            "CORPUS|PRESENT|ok=1|err={}|uploads={}",
            err_of(tree, &ok2),
            dev.writes.len(),
        ));
        dev.print_writes("present", out);
    }
}

// ------------------------------------------------------------------ scenario 5:
// the span checks. The corpus wrote directories that lie about where the bytes
// are, or how many there are, which is the only way to reach these refusals.
fn scenario_spans(tree: &Tree, dev: &mut Dev, out: &mut Vec<String>) {
    for tag in [
        "truncated",
        "overlap",
        "same_offset",
        "no_dims",
        "bad_ne0",
        "bad_type",
        "zero_dim",
        "byte_overflow",
    ] {
        // the reader refused some of these headers; both calls then report it the
        // way the C++ catch around the open does
        let (served, served_err) = match open_shards(tree, &[&format!("span_{tag}.gguf")]) {
            Err(e) => (0, e),
            Ok(gs) => {
                let shards = shards_of(&gs);
                let skip = served_names(&shards, true);
                let base = dev.malloc(65536);
                dev.reg(base, 65536, format!("arena:span_{tag}"));
                dev.arena_base = base;
                let mut wt = WeightTable::new();
                let _ = wt.load(
                    &tree.path(&format!("span_{tag}")),
                    65536,
                    &mut *dev,
                    Some(&skip),
                );
                let mut dense = NativeDense::default();
                let ok = dense.load(&shards, &mut wt, &mut *dev, true, (0, -1), LayerRange::ALL);
                out.push(format!(
                    "CORPUS|SPAN|{tag}|served=1|served_err=|ok={}|err={}",
                    ok.is_ok() as i32,
                    err_of(tree, &ok),
                ));
                continue;
            }
        };
        out.push(format!(
            "CORPUS|SPAN|{tag}|served={served}|served_err={served_err}|ok=0|err={served_err}"
        ));
    }

    // an architecture key that is present and wrong: check_architecture's
    // message comes back verbatim, without the "native dense: " prefix the catch
    // adds to everything else
    let mut out_arch = Vec::new();
    {
        let gs = open_shards(tree, &["arch_wrong.gguf"]).unwrap();
        let shards = shards_of(&gs);
        let base = dev.malloc(65536);
        dev.reg(base, 65536, "arena:arch_wrong".to_string());
        dev.arena_base = base;
        let mut wt = WeightTable::new();
        let _ = wt.load(&tree.path("arch_wrong"), 65536, &mut *dev, None);
        let mut dense = NativeDense::default();
        let ok = dense.load(&shards, &mut wt, &mut *dev, true, (0, -1), LayerRange::ALL);
        out_arch.push(format!(
            "CORPUS|ARCH|wrong|ok={}|err={}",
            ok.is_ok() as i32,
            err_of(tree, &ok),
        ));

        // a shard with no general.architecture and no split keys at all
        let gs2 = open_shards(tree, &["arch_naked.gguf"]).unwrap();
        let shards2 = shards_of(&gs2);
        let mut d2 = NativeDense::default();
        let ok2 = d2.load(&shards2, &mut wt, &mut *dev, true, (0, -1), LayerRange::ALL);
        out_arch.push(format!(
            "CORPUS|ARCH|naked|ok={}|err={}",
            ok2.is_ok() as i32,
            err_of(tree, &ok2),
        ));

        // an unsupported type that is still an eligible name: the `continue` after
        // the table lookup (F32 = 0), and a native name whose matrix is not 2-D
        let gs3 = open_shards(tree, &["unsupported.gguf"]).unwrap();
        let shards3 = shards_of(&gs3);
        let mut d3 = NativeDense::default();
        let ok3 = d3.load(&shards3, &mut wt, &mut *dev, true, (0, -1), LayerRange::ALL);
        out_arch.push(format!(
            "CORPUS|UNSUPPORTED|ok={}|err={}|tensors={}",
            ok3.is_ok() as i32,
            err_of(tree, &ok3),
            d3.tensor_count(),
        ));

        // the same shard loaded twice against one table: the second object sees a
        // reference that already carries native bytes
        let gs4 = open_shards(tree, &["attached.gguf"]).unwrap();
        let shards4 = shards_of(&gs4);
        let base2 = dev.malloc(65536);
        dev.reg(base2, 65536, "arena:attached".to_string());
        dev.arena_base = base2;
        let mut wt2 = WeightTable::new();
        let _ = wt2.load(&tree.path("attached"), 65536, &mut *dev, None);
        let mut first_load = NativeDense::default();
        let one = first_load.load(
            &shards4,
            &mut wt2,
            &mut *dev,
            true,
            (0, -1),
            LayerRange::ALL,
        );
        let mut second_load = NativeDense::default();
        let two = second_load.load(
            &shards4,
            &mut wt2,
            &mut *dev,
            true,
            (0, -1),
            LayerRange::ALL,
        );
        out_arch.push(format!(
            "CORPUS|ATTACHED|first={}|second={}|err={}",
            one.is_ok() as i32,
            two.is_ok() as i32,
            err_of(tree, &two),
        ));

        // the same eligible name in two shards, both architecture-valid
        let gs5 = open_shards(tree, &["dup_a.gguf", "dup_b.gguf"]).unwrap();
        let shards5 = shards_of(&gs5);
        let mut d4 = NativeDense::default();
        let four = d4.load(&shards5, &mut wt, &mut *dev, true, (0, -1), LayerRange::ALL);
        out_arch.push(format!(
            "CORPUS|DUP_NAME|ok={}|err={}",
            four.is_ok() as i32,
            err_of(tree, &four),
        ));
    }
    out.extend(out_arch);
}

// ------------------------------------------------------------------ the replay

#[test]
fn native_dense_corpus() {
    let files = fixtures();
    assert_eq!(files.len(), 100, "the golden embeds 100 fixture files");
    let tree = Tree::new(&files);
    let mut dev = Dev::new();
    let mut out: Vec<String> = Vec::new();

    scenario_bytes(&mut out);
    scenario_fixture(&tree, &mut dev, &mut out, "orca");
    scenario_fixture(&tree, &mut dev, &mut out, "q8");
    scenario_kinds(&tree, &mut dev, &mut out);
    scenario_shards(&tree, &mut dev, &mut out);
    scenario_spans(&tree, &mut dev, &mut out);

    let want = expected();
    let mut bad = 0;
    for i in 0..want.len().max(out.len()) {
        let w = want.get(i).map_or("<none>", |s| s.as_str());
        let g = out.get(i).map_or("<none>", |s| s.as_str());
        if w != g {
            bad += 1;
            if bad <= 500 {
                println!("line {i}\n  cpp:  {w}\n  rust: {g}");
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "{bad} of {} golden lines differ from the Rust replay",
        want.len()
    );
}
