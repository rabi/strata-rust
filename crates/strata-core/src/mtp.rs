//! `src/core/mtp.cpp`'s deterministic core: the runtime index, the required-tensor gate, the K/V ring
//! decision, the arena carve, and `bind_bytes`.
//!
//! The CUDA seam is injected (the loader counts its allocations and the nth one can be made to fail), so the
//! layout is the real code's answer without a device.  Four size helpers are transcribed from `.cu` files —
//! each cites the line — because the corpus builds without nvcc; a misread of one of those is invisible to
//! the replay.

use crate::layer::{
    qsa_decode_attn_scratch_floats, qsa_shapes, qsa_state_bytes, shared_expert_scratch_bytes,
    KvFlags,
};
use crate::layout::ModelGeometry;

/// `include/strata/kernels/cpu/expert.hpp:45`.
pub const BLOB: u64 = 1_382_400;
/// `verify_kernels.hpp:21`.
pub const K_VERIFY_MAX_T: i32 = 8;
/// The penalty ring's capacity, `coupled_draft.hpp`.
pub const K_COUPLED_HIST_CAP: i64 = 4096;

/// `native_mmvq.cu:1250` (Q8K = 32, sizeof(Q81Block) = 36).
pub fn native_q8_1_bytes(n_in: i32, ncols: i32) -> u64 {
    ncols as u64 * (n_in as i64 / 32) as u64 * 36
}

/// `s2_expert_grouped.cu:568`.
pub fn moe_hit_grouped_scratch_bytes(n_hits: i64, n_embd: i64, n_ff: i64) -> u64 {
    if n_hits <= 0 {
        return 0;
    }
    let al = |v: u64| (v + 15) & !15u64;
    let gu = n_hits as u64 * (2 * n_ff) as u64 * 4;
    let q8 = n_hits as u64 * (n_ff / 32) as u64 * 34;
    let hs = n_hits as u64 * (n_ff / 32) as u64 * 4;
    let xh = (n_embd / 32) as u64 * 4;
    al(gu) + al(q8) + 2 * al(hs) + al(xh)
}

/// `sampler.cu:1037,1048` (kSplitBlockSpan = 4096, kSplitMaxBlocks = 64, kSelMax = 64, sizeof(int2) = 8).
pub fn coupled_draft_scratch_bytes(nv: i32) -> u64 {
    let blocks = (nv as i64 + 4096 - 1) / 4096;
    if nv <= 0 || blocks > 64 {
        return 0;
    }
    blocks as u64 * 64 * 8
}

/// `mtp.cpp:57` — 256-byte steps, and a null base hands back null.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bump {
    pub base: Option<u64>,
    pub used: u64,
}

impl Bump {
    pub fn new(base: Option<u64>) -> Self {
        Bump { base, used: 0 }
    }

    /// `take<T>(n)`: hand back the current position, then step over the aligned region.
    pub fn take(&mut self, count: u64, size: u64) -> Option<u64> {
        let r = self.base.map(|b| b + self.used);
        self.used += (count.wrapping_mul(size) + 255) & !255u64;
        r
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub kind: String,
    pub rows: i64,
    pub cols: i64,
    pub off: u64,
    pub bytes: u64,
}

/// The nine the loader refuses without.  Order decides which message fires first.
pub const REQUIRED_Q8: [&str; 9] = [
    "fc_embedding.weight",
    "fc_hidden.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
];

/// `operator>>` for a signed 64-bit field: skip whitespace, optional sign, digits, stop at the first
/// non-digit; no digits at all fails the stream.  Out of range wraps rather than failing.
fn stream_i64(s: &str) -> Option<(i64, String)> {
    let t = s.trim_start();
    let (neg, rest) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let v = digits.parse::<u128>().unwrap_or(u128::MAX) as i128;
    let v = if neg { -v } else { v };
    Some((v as i64, rest[digits.len()..].to_string()))
}

fn field(s: &str) -> Option<(String, String)> {
    let t = s.trim_start();
    if t.is_empty() {
        return None;
    }
    match t.find(char::is_whitespace) {
        Some(i) => Some((t[..i].to_string(), t[i..].to_string())),
        None => Some((t.to_string(), String::new())),
    }
}

/// `dense.txt`: `name kind rows cols off bytes`.  Empty lines are skipped; a field the stream cannot fill
/// fails the line.
pub fn parse_index(text: &str) -> Result<Vec<Tensor>, String> {
    let mut out = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let mut rest = line.to_string();
        let mut t = Tensor::default();
        for i in 0..6 {
            let (raw, tail) = field(&rest).ok_or_else(|| line.to_string())?;
            rest = tail;
            match i {
                0 => t.name = raw,
                1 => t.kind = raw,
                2 => t.rows = stream_i64(&raw).ok_or_else(|| line.to_string())?.0,
                3 => t.cols = stream_i64(&raw).ok_or_else(|| line.to_string())?.0,
                4 => t.off = stream_i64(&raw).ok_or_else(|| line.to_string())?.0 as u64,
                _ => t.bytes = stream_i64(&raw).ok_or_else(|| line.to_string())?.0 as u64,
            }
        }
        out.push(t);
    }
    Ok(out)
}

/// The lookup the loader runs: name AND kind must match, which is why a tensor listed as f32 is not found by
/// the q8_0 gate.
pub fn find<'a>(tensors: &'a [Tensor], name: &str, kind: &str) -> Option<&'a Tensor> {
    tensors.iter().find(|t| t.name == name && t.kind == kind)
}

/// `bind_bytes` (`mtp.cpp:313`).  `head_logits_bound`/`dhead_bound` are the bindings the caller has already
/// made; `draft_vocab_len` is `draft_vocab.bin`'s size when the file opens.
pub fn bind_bytes(
    max_t: i32,
    n_vocab: i64,
    head_logits_bound: bool,
    dhead_bound: bool,
    draft_vocab_len: Option<u64>,
    head_row_bytes: u64,
    coupled: bool,
) -> u64 {
    let mut bytes = if head_logits_bound {
        0
    } else {
        max_t as u64 * n_vocab as u64 * 4
    };
    if !dhead_bound {
        if let Some(size) = draft_vocab_len {
            if size >= 4 && size % 4 == 0 {
                bytes += (size / 4) * head_row_bytes + size;
            }
        }
    }
    if coupled {
        bytes += n_vocab as u64 * 4
            + ((K_COUPLED_HIST_CAP + max_t as i64) as u64) * 4
            + 64 * 64 * 8
            + 4096;
    }
    bytes
}

/// Everything `load` decides, in the order it decides it.  `err` is the loader's message when it stops
/// early; `vram`, `kv_mode` and `allocs` are whatever had accumulated at that point.
#[derive(Debug)]
pub struct Run {
    pub err: Option<String>,
    pub vram: u64,
    pub kv_mode: i32,
    pub allocs: Vec<(u64, u64)>,
    pub fields: Vec<(&'static str, Option<u64>)>,
    pub cap: i64,
    pub bind: u64,
}

/// The inputs `load` reads from the outside world.
#[derive(Debug)]
pub struct Inputs<'a> {
    pub index: Option<&'a str>,
    pub dense_bin_len: Option<u64>,
    pub experts_bin_len: Option<u64>,
    pub draft_vocab_len: Option<u64>,
    pub max_t: i32,
    pub window: i64,
    pub max_cells: i64,
    pub k: i64,
    pub head_row_bytes: u64,
    pub n_vocab: i64,
    pub coupled: bool,
    /// `qsa_set_kv_resident`: 0 keeps every cell in VRAM.
    pub resident: i64,
    /// Which `cudaMalloc` fails, 1-based: dense, experts, state, arena.
    pub fail_malloc_at: usize,
}

/// The carve, in the C++'s order, with the element size each `take<T>` uses.
const CARVE: &[(&str, u64)] = &[
    ("tok", 4),
    ("step", 4),
    ("pos", 4),
    ("row", 4),
    ("ident", 4),
    ("Rin", 4),
    ("R", 4),
    ("emb", 4),
    ("en", 4),
    ("e2", 4),
    ("hn", 4),
    ("h2", 4),
    ("mixed", 4),
    ("inj", 4),
    ("inj2", 4),
    ("lo", 4),
    ("rs", 4),
    ("bo", 4),
    ("xn", 4),
    ("xq", 1),
    ("qfull", 4),
    ("qcur", 4),
    ("kcur", 4),
    ("vcur", 4),
    ("attn", 4),
    ("attn32", 4),
    ("attn_scratch", 4),
    ("logits", 4),
    ("w", 4),
    ("ids", 4),
    ("shared", 4),
    ("parts", 4),
    ("y", 4),
    ("sample", 4),
    ("hit_slot", 4),
    ("hit_dst", 4),
    ("hit_count", 4),
    ("grp_ptr", 8),
    ("grp_start", 4),
    ("grp_counts", 4),
    ("hit_xq", 1),
    ("hit_xs", 4),
    ("hit_scratch", 1),
    ("sh_scratch", 1),
    ("x_bf16", 2),
    ("out_ids", 4),
    ("probs", 4),
    ("dummy_inj", 4),
];

fn carve(
    g: &ModelGeometry,
    t: u64,
    r2: u64,
    k: u64,
    cap: u64,
    scratch: u64,
) -> (Vec<(&'static str, Option<u64>)>, u64) {
    let n = g.n_embd as u64;
    let hc = g.hc as u64;
    let nh = g.n_head as u64;
    let hd = g.head_dim as u64;
    let nk = g.n_head_kv as u64;
    let counts: [u64; 48] = [
        t,
        r2 * 4,
        r2 * nh,
        4,
        t * cap,
        t * hc * n,
        t * hc * n,
        t * n,
        t * n,
        t * n,
        t * hc * n,
        t * hc * n,
        t * n,
        t * hc,
        t * hc,
        t * g.hc_lr as u64,
        t * hc,
        t * n,
        t * hc * n,
        native_q8_1_bytes((nh * hd) as i32, 8),
        t * nh * 2 * hd,
        t * nh * hd,
        t * nk * hd,
        t * nk * hd,
        t * nh * hd,
        t * nh * hd,
        scratch,
        t * g.n_expert as u64,
        t * k,
        t * k,
        t * n,
        t * k * n,
        t * n,
        t * n,
        t * k,
        t * k,
        4,
        t * k,
        t * k + 1,
        4,
        t * (n / 32) * 34,
        t * (n / 32),
        moe_hit_grouped_scratch_bytes((t * k) as i64, g.n_embd, g.n_ff),
        shared_expert_scratch_bytes(g.n_ff),
        n,
        t + 4,
        t + 4,
        hc,
    ];
    let mut b = Bump::new(Some(0));
    let mut out = Vec::new();
    for ((name, size), count) in CARVE.iter().zip(counts.iter()) {
        out.push((*name, b.take(*count, *size)));
    }
    let used = b.used;
    (out, used)
}

/// `MtpDrafter::load`'s deterministic path.
pub fn load(g: &ModelGeometry, inp: &Inputs, root: &str) -> Run {
    let mut run = Run {
        err: None,
        vram: 0,
        kv_mode: 0,
        allocs: Vec::new(),
        fields: Vec::new(),
        cap: 0,
        bind: 0,
    };
    let mut slab = 0u64;
    let mut malloc = |n: u64, run: &mut Run| {
        if inp.fail_malloc_at == run.allocs.len() + 1 {
            return false;
        }
        run.allocs.push((slab, n));
        slab += n;
        true
    };

    if inp.max_t < 1 || inp.max_t > K_VERIFY_MAX_T {
        run.err = Some("mtp: max_t out of range".into());
        return run;
    }
    let text = match inp.index {
        Some(t) => t,
        None => {
            run.err = Some(format!(
                "mtp: cannot open {root}/dense.txt (run tools/mtp_rt.py)"
            ));
            return run;
        }
    };
    let tensors = match parse_index(text) {
        Ok(v) => v,
        Err(line) => {
            run.err = Some(format!("mtp: malformed dense.txt line: {line}"));
            return run;
        }
    };
    let dense_len = match inp.dense_bin_len {
        Some(n) => n,
        None => {
            run.err = Some("mtp: cannot read dense.bin".into());
            return run;
        }
    };
    if !malloc(dense_len, &mut run) {
        run.err = Some(format!(
            "mtp: dense weights allocation failed (stub), requested {} MiB, CUDA0 free {} MiB",
            dense_len >> 20,
            1024
        ));
        return run;
    }
    run.vram += dense_len;

    let experts_bytes = g.n_expert as u64 * BLOB;
    if inp.experts_bin_len.is_none() {
        run.err = Some("mtp: cannot open experts.bin".into());
        return run;
    }
    if !malloc(experts_bytes, &mut run) {
        run.err = Some("mtp: the 512 experts do not fit in VRAM".into());
        return run;
    }
    run.vram += experts_bytes;

    for name in REQUIRED_Q8 {
        if find(&tensors, name, "q8_0").is_none() {
            run.err = Some(format!("mtp: {name} is missing (q8_0)"));
            return run;
        }
    }

    let s = qsa_shapes(g);
    let max_cells = inp.max_cells;
    let ring = if inp.window > 0 && inp.window < max_cells {
        inp.window + 4 * inp.max_t as i64 + 64
    } else {
        0
    };
    // the drafter forces hybrid off, and int8 on only when the model was started with k8v4
    let flags = KvFlags {
        resident: inp.resident,
        ..KvFlags::default()
    };
    let sb = qsa_state_bytes(g, max_cells, false, ring, &flags);
    if !malloc(sb, &mut run) {
        run.err = Some("mtp: the K/V state does not fit".into());
        return run;
    }
    run.kv_mode = crate::layer::kv_plan(&s, max_cells, ring, &flags).mode;
    run.vram += sb;

    let window = if inp.window > 0 && inp.window < max_cells {
        inp.window
    } else {
        0
    };
    let cap = ((if window > 0 { window } else { max_cells }) + 63) / 64 * 64;
    let scratch = qsa_decode_attn_scratch_floats(cap, &s);
    let t = inp.max_t as u64;
    let (fields, arena_used) = carve(g, t, 2 * t, inp.k as u64, cap as u64, scratch);
    if !malloc(arena_used, &mut run) {
        run.err = Some("mtp: buffers do not fit".into());
        return run;
    }
    run.vram += arena_used;

    run.fields = fields;
    run.cap = cap;
    run.bind = bind_bytes(
        inp.max_t,
        inp.n_vocab,
        false,
        false,
        inp.draft_vocab_len,
        inp.head_row_bytes,
        inp.coupled,
    );
    run
}
