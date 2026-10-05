//! `src/core/layer.cpp`'s deterministic surface: the canonical-form gate, the plane gate, the arena
//! layouts, and the KV streaming plan.
//!
//! Pointers become offsets.  `Cursor` is the same 16-byte bump allocator, `Planes` carries offsets into the
//! tensor region instead of `const uint8_t*`, and every `*_init` returns the region offsets in field order
//! plus the total the C++ returns.
//!
//! Four size helpers live in `.cu` files (`shared_expert_scratch_bytes`, `qsa_decode_attn_scratch_floats`,
//! `kv_block_bytes`, `gr_workspace_init`).  The corpus re-declares them from the source rather than building
//! the kernels, so those four are transcribed on both sides and the corpus cannot catch a misread of them.
//! Everything else here is checked against the real C++.

use crate::layout::ModelGeometry;
use crate::weights::WeightRef;

pub const Q8K_BYTES_PER_BLOCK: u64 = 292;
pub const Q8K_ELEMS_PER_BLOCK: i64 = 256;

/// A Q8_K buffer needs `n` a multiple of 256 and `n/256` blocks of 292 bytes.
pub fn q8k_bytes(n: i64) -> u64 {
    (n / Q8K_ELEMS_PER_BLOCK) as u64 * Q8K_BYTES_PER_BLOCK
}

pub fn align_up16(n: u64) -> u64 {
    (n + 15) & !15
}

/// One cursor over an arena, so every region is 16-byte aligned without a list of hand-added offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cursor {
    used: u64,
}

impl Cursor {
    pub fn new() -> Self {
        Cursor { used: 0 }
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    /// `take<T>(count)` in the C++: hand back the current offset, then step over the aligned region.
    pub fn take(&mut self, count: u64, size: u64) -> u64 {
        let r = self.used;
        self.used = align_up16(self.used.wrapping_add(count.wrapping_mul(size)));
        r
    }

    pub fn take_bytes(&mut self, n: u64) -> u64 {
        self.take(n, 1)
    }
}

impl Default for Cursor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codebook {
    Affine = 0,
    Iq4Nl = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SForm {
    pub code_bits: i32,
    pub code_bias: i32,
    pub group_elems: i32,
    pub codebook: Codebook,
    pub has_offset: bool,
    /// Carried, not derived — see the note on `SForm::act_kind` in `s_gemv.hpp`.
    pub act_kind: i32,
}

/// `s_gemv_q8k` takes the canonical-form attributes; a tensor that is NOT quantized has none, and building a
/// form out of zeroes would decode every code as `0 + bias` and produce a perfectly finite wrong answer.
pub fn sform_of(r: &WeightRef, name: &str) -> Result<SForm, String> {
    if !r.quantized() {
        return Err(format!(
            "{name} is not a quantized tensor, so it has no S-form"
        ));
    }
    Ok(SForm {
        code_bits: r.code_bits,
        code_bias: r.code_bias,
        group_elems: r.group_elems,
        codebook: if r.codebook_iq4nl {
            Codebook::Iq4Nl
        } else {
            Codebook::Affine
        },
        has_offset: r.has_offset,
        act_kind: r.act_kind,
    })
}

/// The three canonical planes of a quantized tensor, as offsets INSIDE the loaded region.
///
/// This does not re-derive the layout — the sizes come from the index and this only checks that they describe
/// the tensor it was given.  Re-deriving them is how a width bug survived a round (see the comment at
/// `layer.cpp:70`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Planes {
    pub codes: u64,
    pub scales: u64,
    /// `None` when the form has none.
    pub offset: Option<u64>,
}

pub fn plane_ptrs(r: &WeightRef, name: &str) -> Result<Planes, String> {
    // S2, S4 and S8 all split the same way; the guard catches a tensor that is not quantized at all.
    if r.code_bits != 2 && r.code_bits != 4 && r.code_bits != 8 {
        return Err(format!(
            "{name}: code_bits {} is not an S2/S4/S8 form",
            r.code_bits
        ));
    }
    let end = r.codes_bytes + r.scales_bytes + r.offset_bytes;
    if r.codes_bytes == 0 || r.scales_bytes == 0 || end != r.bytes {
        return Err(format!(
            "{name}: the planes add up to {end} B but the tensor is {} B (codes {}, scales {}, \
             offsets {}) - what the loader recorded is not the layout that was loaded",
            r.bytes, r.codes_bytes, r.scales_bytes, r.offset_bytes
        ));
    }
    if r.has_offset != (r.offset_bytes != 0) {
        return Err(format!(
            "{name}: has_offset is {} but the offset plane is {} B",
            r.has_offset as u8, r.offset_bytes
        ));
    }
    Ok(Planes {
        codes: 0,
        scales: r.codes_bytes,
        offset: if r.offset_bytes != 0 {
            Some(r.codes_bytes + r.scales_bytes)
        } else {
            None
        },
    })
}

// ---------------------------------------------------------------- the QSA shapes

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QsaShapes {
    pub n_head: i64,
    pub n_head_kv: i64,
    pub head_dim: i64,
    pub n_rot: i64,
    pub idx_n_head: i64,
    pub idx_dim: i64,
    pub idx_block: i64,
    pub idx_top_k: i64,
    pub page_size: i64,
}

pub fn qsa_real_shapes() -> QsaShapes {
    QsaShapes {
        n_head: 24,
        n_head_kv: 2,
        head_dim: 256,
        n_rot: 64,
        idx_n_head: 4,
        idx_dim: 128,
        idx_block: 4,
        idx_top_k: 2048,
        page_size: 4,
    }
}

/// The geometry the QSA kernels want, from the one place that defines it.
pub fn qsa_shapes(g: &ModelGeometry) -> QsaShapes {
    let mut s = qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    s
}

pub const K_TOPK_MAX_CELLS: i64 = 32768;

pub fn qsa_selection_width(n_kv: i64, s: &QsaShapes) -> i64 {
    let w = s.idx_top_k + s.idx_block - 1;
    if n_kv < w {
        n_kv
    } else {
        w
    }
}

pub const KV_Q8_GROUP: i64 = 64;
pub const QK4_0: i64 = 32;
pub const BLOCK_Q4_0_BYTES: u64 = 18;
pub const K_KV_CTL_INTS: u64 = 16;
pub const K_STEP_COUNT: u64 = 4;
/// `qsa_decode_attn.cu`'s chunk width and head width.
pub const QSA_DECODE_CHUNK: i64 = 64;
pub const QSA_DECODE_HD: i64 = 256;

pub const KV_F16: i32 = 0;
pub const KV_INT8: i32 = 1;
pub const KV_Q4: i32 = 2;

pub fn kv_q4_bytes_per_head(head_dim: i64) -> u64 {
    (head_dim / QK4_0) as u64 * BLOCK_Q4_0_BYTES
}

pub fn kv_q4_bytes_per_cell(s: &QsaShapes) -> u64 {
    s.n_head_kv as u64 * kv_q4_bytes_per_head(s.head_dim) * 2
}

pub fn kv_q8_bytes_per_cell(s: &QsaShapes) -> u64 {
    s.n_head_kv as u64 * s.head_dim as u64 * 2
        + s.n_head_kv as u64 * (s.head_dim / KV_Q8_GROUP) as u64 * 2 * 2
}

pub fn kv_block_bytes(s: &QsaShapes, fmt: i32) -> u64 {
    let rows = (s.n_head_kv * s.page_size) as u64;
    if fmt == KV_Q4 {
        return rows * kv_q4_bytes_per_head(s.head_dim) * 2;
    }
    if fmt == KV_INT8 {
        rows * s.head_dim as u64 * 2 + rows * (s.head_dim / KV_Q8_GROUP) as u64 * 2 * 2
    } else {
        rows * s.head_dim as u64 * 2 * 2
    }
}

pub fn kv_stream_map_bytes(n_slots: i64) -> u64 {
    n_slots as u64 * 4 * 5 + K_KV_CTL_INTS * 4
}

pub fn qsa_step_bytes() -> u64 {
    4 * K_STEP_COUNT
}

pub fn shared_expert_scratch_bytes(n_ff: i64) -> u64 {
    // gate (n_ff f32) | up (n_ff f32) | q8_0 (n_ff/32*34) | q8k (n_ff/256*292) | g (1 f32), 16-byte aligned
    let a = align_up16(n_ff as u64 * 4);
    let q0 = align_up16(((n_ff / 32) * 34) as u64);
    let qk = align_up16(((n_ff / 256) * 292) as u64);
    a * 2 + q0 + qk + 32
}

pub fn qsa_decode_attn_scratch_floats(cap: i64, s: &QsaShapes) -> u64 {
    let chunks = (cap + QSA_DECODE_CHUNK - 1) / QSA_DECODE_CHUNK;
    chunks as u64 * s.n_head as u64 * (QSA_DECODE_HD as u64 + 2) + 64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrShapes {
    pub n_embd: i64,
    pub hc: i64,
    pub hc_lr: i64,
}

/// One table, indexed by the same `k` that assigns the offsets below — a mis-ordered table is what let two
/// regions overlap for a whole round (`gr.cu:306`).
pub fn gr_workspace_init(s: &GrShapes, carve: bool) -> (u64, Vec<(&'static str, u64)>) {
    let hc_dim = (s.hc * s.n_embd) as u64;
    let hc_lr = s.hc_lr as u64;
    let sz = [
        hc_dim * 4, // xn
        hc_dim * 2, // xq
        hc_lr * 2,  // lq
        hc_dim * 4, // gated
        hc_lr * 4,  // lo
    ];
    const NAMES: [&str; 5] = ["xn", "xq", "lq", "gated", "lo"];
    let mut al = [0u64; 5];
    let mut bytes = 0u64;
    for k in 0..5 {
        al[k] = align_up16(sz[k]);
        bytes += al[k];
    }
    let mut out = Vec::new();
    if carve {
        let mut p = 0u64;
        for k in 0..5 {
            out.push((NAMES[k], p));
            p += al[k];
        }
    }
    (bytes, out)
}

pub fn gr_workspace_bytes(s: &GrShapes) -> u64 {
    gr_workspace_init(s, false).0
}

// ---------------------------------------------------------------- the KV plan

/// The process-wide switches `kv_plan` reads, injected so the corpus can drive them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvFlags {
    pub resident: i64,
    pub int8: bool,
    pub q4: bool,
    pub hybrid: bool,
    pub ring_off: bool,
    pub main_off: bool,
}

pub fn qsa_kv_resident_min() -> i64 {
    20480
}

/// How one state holds its K/V: `mode` as in `QsaState::kv_mode`, `slots` VRAM pages of `pages` logical ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPlan {
    pub mode: i32,
    pub pages: i64,
    pub slots: i64,
    pub pooled_rows: i64,
}

pub fn kv_plan(s: &QsaShapes, max_cells: i64, ring_cells: i64, f: &KvFlags) -> KvPlan {
    let mut p = KvPlan {
        mode: 0,
        pages: (max_cells + s.page_size - 1) / s.page_size,
        slots: 0,
        pooled_rows: 0,
    };
    p.slots = p.pages;
    p.pooled_rows = max_cells / s.idx_block + 2;
    if f.resident <= 0 || ring_cells < 0 {
        return p; // ring_cells < 0: always fully resident
    }
    if (ring_cells > 0 && f.ring_off) || (ring_cells <= 0 && f.main_off) {
        return p;
    }
    if ring_cells > 0 {
        let r = (ring_cells + s.page_size - 1) / s.page_size;
        if r < p.pages {
            p.mode = 2;
            p.slots = r;
            p.pooled_rows = 2; // the drafter has no indexer
        }
    } else {
        let r = (std::cmp::max(f.resident, qsa_kv_resident_min()) + s.page_size - 1) / s.page_size;
        if r < p.pages {
            p.mode = 1;
            p.slots = r;
        }
    }
    p
}

pub fn kv_pool_bytes(s: &QsaShapes, pages: i64, hybrid: bool, int8: bool, f: &KvFlags) -> u64 {
    if hybrid {
        // K8V4: the INT8 K half (codes + scales) plus the Q4_0 V half (kv_q4.hpp's rotation)
        let rows = (pages * s.page_size * s.n_head_kv) as u64;
        return rows * s.head_dim as u64
            + rows * (s.head_dim / KV_Q8_GROUP) as u64 * 2
            + rows * kv_q4_bytes_per_head(s.head_dim)
            + 64;
    }
    if f.q4 {
        return (pages * s.page_size) as u64 * kv_q4_bytes_per_cell(s) + 64;
    }
    if int8 {
        (pages * s.page_size) as u64 * kv_q8_bytes_per_cell(s) + 64
    } else {
        (pages * s.page_size * s.n_head_kv * s.head_dim * 2 * 2) as u64
    }
}

// ---------------------------------------------------------------- arena layouts

type Layout = (Vec<(&'static str, u64)>, u64);

/// Parts summed with each one aligned up — the C++ `for (uint64_t v : parts) total += (v + 15) & ~15ull`.
fn aligned_total(parts: &[u64]) -> u64 {
    parts.iter().map(|v| align_up16(*v)).sum()
}

fn named(parts: &[u64], names: &[&'static str]) -> Vec<(&'static str, u64)> {
    parts
        .iter()
        .zip(names)
        .map(|(n, name)| (*name, *n))
        .collect()
}

const GDN_FIELDS: [&str; 16] = [
    "x_q8k",
    "x_q8_0",
    "x_bf16",
    "qkv",
    "conv_out",
    "h",
    "alpha",
    "beta",
    "gate",
    "o",
    "z",
    "y",
    "y_q8k",
    "y_q8_0",
    "state",
    "conv_state",
];

fn gdn_parts(g: &ModelGeometry) -> [u64; 16] {
    let c = g.ssm_conv_channels;
    let v = g.ssm_value_dim;
    [
        q8k_bytes(g.n_embd),
        ((g.n_embd / 32) * 34) as u64,
        (g.n_embd * 2) as u64,
        (c * 4) as u64,
        (c * 4) as u64,
        (c * 4) as u64,
        (g.ssm_v_heads * 4) as u64,
        (g.ssm_v_heads * 4) as u64,
        (g.ssm_v_heads * 4) as u64,
        (g.ssm_v_heads * g.ssm_state_size * 4) as u64,
        (v * 4) as u64,
        (v * 4) as u64,
        q8k_bytes(v),
        ((v / 32) * 34) as u64,
        (g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size * 4) as u64,
        (c * (g.ssm_d_conv - 1) * 4) as u64,
    ]
}

pub fn gdn_buffers_bytes(g: &ModelGeometry) -> u64 {
    aligned_total(&gdn_parts(g))
}

pub fn gdn_buffers_init(g: &ModelGeometry) -> Layout {
    carve(&named(&gdn_parts(g), &GDN_FIELDS))
}

const MOE_FIELDS: [&str; 9] = [
    "x_bf16",
    "x_f16",
    "logits",
    "ids",
    "weights",
    "shared",
    "sh_scratch",
    "x_q8_0",
    "x_q8k",
];

fn moe_parts(g: &ModelGeometry, k: i64) -> [u64; 9] {
    [
        (g.n_embd * 2) as u64,
        (g.n_embd * 2) as u64,
        (g.n_expert * 4) as u64,
        (k * 4) as u64,
        (k * 4) as u64,
        (g.n_embd * 4) as u64,
        shared_expert_scratch_bytes(g.n_ff),
        ((g.n_embd / 32) * 34) as u64,
        q8k_bytes(g.n_embd),
    ]
}

pub fn moe_buffers_bytes(g: &ModelGeometry, k: i64) -> u64 {
    aligned_total(&moe_parts(g, k))
}

pub fn moe_buffers_init(g: &ModelGeometry, k: i64) -> Layout {
    carve(&named(&moe_parts(g, k), &MOE_FIELDS))
}

/// `qsa_buffers_bytes` sums the parts RAW — no per-part alignment — and then aligns the total and adds 256.
/// `qsa_buffers_init` runs a Cursor, which aligns every part.  The two agree only when every part is already
/// 16-aligned; the corpus pins where they do not.
pub fn qsa_buffers_bytes(g: &ModelGeometry, max_cells: i64) -> u64 {
    let s = qsa_shapes(g);
    let cap = qsa_selection_width(K_TOPK_MAX_CELLS, &s);
    let mut n = 0u64;
    n += q8k_bytes(g.n_embd);
    n += ((g.n_embd / 32) * 34) as u64;
    n += (g.n_embd * 2) as u64;
    n += (g.n_head * 2 * g.head_dim * 4) as u64;
    n += (g.n_head * g.head_dim * 4) as u64;
    n += (g.n_head_kv * g.head_dim * 4 * 2) as u64;
    n += (g.idx_key_dim * 4) as u64;
    n += (g.idx_q_heads * g.idx_key_dim * 4) as u64;
    n += (max_cells * 4) as u64;
    n += (cap * 4) as u64;
    n += (cap * g.n_head_kv * g.head_dim * 2 * 2) as u64;
    n += (g.n_head * g.head_dim * 4) as u64;
    n += (g.n_head * g.head_dim * 2) as u64;
    n += (g.n_head * g.head_dim * 4) as u64;
    n += q8k_bytes(g.n_head * g.head_dim);
    n += qsa_decode_attn_scratch_floats(cap, &s) * 4 + 16;
    align_up16(n) + 256
}

pub fn qsa_buffers_init(g: &ModelGeometry, max_cells: i64) -> Layout {
    let s = qsa_shapes(g);
    let cap = qsa_selection_width(K_TOPK_MAX_CELLS, &s);
    let parts: [(&str, u64); 18] = [
        ("x_q8k", q8k_bytes(g.n_embd)),
        ("x_q8_0", ((g.n_embd / 32) * 34) as u64),
        ("x_bf16", (g.n_embd * 2) as u64),
        ("q_full", (g.n_head * 2 * g.head_dim * 4) as u64),
        ("qcur", (g.n_head * g.head_dim * 4) as u64),
        ("kcur", (g.n_head_kv * g.head_dim * 4) as u64),
        ("vcur", (g.n_head_kv * g.head_dim * 4) as u64),
        ("idx_raw", (g.idx_key_dim * 4) as u64),
        ("q_idx", (g.idx_q_heads * g.idx_key_dim * 4) as u64),
        ("cell_scores", (max_cells * 4) as u64),
        ("ids", (cap * 4) as u64),
        ("k_scratch", (cap * g.n_head_kv * g.head_dim * 2) as u64),
        ("v_scratch", (cap * g.n_head_kv * g.head_dim * 2) as u64),
        ("attn", (g.n_head * g.head_dim * 4) as u64),
        ("attn16", (g.n_head * g.head_dim * 2) as u64),
        ("attn32", (g.n_head * g.head_dim * 4) as u64),
        ("attn_q8k", q8k_bytes(g.n_head * g.head_dim)),
        ("attn_scratch", qsa_decode_attn_scratch_floats(cap, &s) * 4),
    ];
    carve(&parts)
}

pub fn block_buffers_bytes(g: &ModelGeometry) -> u64 {
    let s = GrShapes {
        n_embd: g.n_embd,
        hc: g.hc,
        hc_lr: g.hc_lr,
    };
    let mut n = 0u64;
    n += (g.hc * g.n_embd * 4) as u64;
    n += (g.n_embd * 4) as u64;
    n += (g.n_embd * 4) as u64;
    n += (g.hc * 4) as u64;
    n += (g.hc * 4 * 2 + 32) as u64;
    n += q8k_bytes(g.n_embd);
    n += gr_workspace_bytes(&s);
    align_up16(n) + 256
}

pub fn block_buffers_init(g: &ModelGeometry) -> Layout {
    let s = GrShapes {
        n_embd: g.n_embd,
        hc: g.hc,
        hc_lr: g.hc_lr,
    };
    let parts: [(&str, u64); 8] = [
        ("R", (g.hc * g.n_embd * 4) as u64),
        ("mixed", (g.n_embd * 4) as u64),
        ("block_out", (g.n_embd * 4) as u64),
        ("inject", (g.hc * 4) as u64),
        ("inject2", (g.hc * 4) as u64),
        ("gr_rs", (g.hc * 4) as u64),
        ("head_q8k", q8k_bytes(g.n_embd)),
        ("gr", gr_workspace_bytes(&s)),
    ];
    let (mut out, used) = carve(&parts);
    let grw = off(&out, "gr");
    out.retain(|(n, _)| *n != "gr");
    for (name, o) in gr_workspace_init(&s, true).1 {
        out.push((name, o + grw));
    }
    (out, used)
}

pub fn qsa_state_bytes(
    g: &ModelGeometry,
    max_cells: i64,
    with_rope: bool,
    ring_cells: i64,
    f: &KvFlags,
) -> u64 {
    let s = qsa_shapes(g);
    let p = kv_plan(&s, max_cells, ring_cells, f);
    let mut n = 0u64;
    n += kv_pool_bytes(
        &s,
        p.slots,
        f.hybrid && ring_cells <= 0,
        f.int8 || f.hybrid,
        f,
    ) + 4 * 16;
    n += (p.pages * 4) as u64;
    if p.mode == 1 {
        n += kv_stream_map_bytes(p.slots) + 6 * 16;
    }
    n += ((s.idx_block - 1) * s.idx_dim * 4) as u64;
    n += (s.idx_dim * 4) as u64;
    n += (p.pooled_rows * s.idx_dim * 4) as u64;
    n += 16;
    if with_rope {
        n += (max_cells * (s.n_rot / 2) * 4 * 2) as u64;
    }
    n += qsa_step_bytes() + 16;
    n += (s.n_head * 4) as u64;
    align_up16(n) + 256
}

/// Per-layer dump stride, in floats.
pub fn dump_stride_floats(g: &ModelGeometry) -> u64 {
    let nvk = (g.n_head_kv * g.head_dim) as u64;
    (2 * g.n_embd) as u64 + (2 * g.hc) as u64 + (g.n_head * g.head_dim) as u64 + 5 * nvk + 8
}

fn carve(parts: &[(&'static str, u64)]) -> Layout {
    let mut c = Cursor::new();
    let mut out: Vec<(&str, u64)> = Vec::with_capacity(parts.len());
    for (name, n) in parts {
        out.push((*name, c.take_bytes(*n)));
    }
    let used = c.used();
    (out, used)
}

fn off(layout: &[(&'static str, u64)], name: &str) -> u64 {
    layout
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, o)| *o)
        .unwrap()
}
