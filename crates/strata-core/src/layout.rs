//! Port of `src/core/layout.cpp` — the shape checks.
//!
//! `WeightLookup` answers "where are the bytes"; this answers "is this the tensor I think it
//! is". Two real incidents make the case:
//!
//! * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY
//!   heads. The cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so
//!   the term was 4x too big — and the error survived a round because it moved in the
//!   direction that TIGHTENS the budget.
//! * `docs/semantics.md` and the tensor manifest agree on every dimension, but nothing
//!   CHECKED that a named tensor had the shape the kernel reading it assumes.
//!
//! So this module names the tensors per layer type and asserts every shape a kernel depends
//! on, at LOAD time: a mismatch is reported with the tensor name, the shape found and the
//! shape required, not as a wrong number inside a GEMV at token 4000.
//!
//! The `WeightRef` here carries only the fields a shape check reads. The loader port adds the
//! rest (source provenance, plane sizes, activation kind) — a caller that has a full loader
//! `WeightRef` can project it onto this view.

use Family::{Either, Gdn, Qsa};

/// The model's geometry, from `docs/semantics.md` and the artifact's own metadata. Every
/// field is a number a kernel depends on, so a change is a change to a kernel contract and
/// not a tuning knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelGeometry {
    pub n_embd: i64,
    pub n_layers: i64,
    /// every 4th layer is full attention: layers 3, 7, ... 47
    pub qsa_interval: i64,
    // GDN (36 layers)
    pub ssm_state_size: i64,
    pub ssm_k_heads: i64,
    pub ssm_v_heads: i64,
    pub ssm_d_conv: i64,
    /// 2*128*16 + 128*48
    pub ssm_conv_channels: i64,
    /// 128 * 48
    pub ssm_value_dim: i64,
    // QSA (12 layers)
    pub n_head: i64,
    pub n_head_kv: i64,
    pub head_dim: i64,
    pub idx_q_heads: i64,
    pub idx_key_dim: i64,
    // gated residual, on every layer
    pub hc: i64,
    pub hc_lr: i64,
    // MoE, on every layer
    pub n_expert: i64,
    pub n_ff: i64,
}

impl Default for ModelGeometry {
    fn default() -> Self {
        ModelGeometry {
            n_embd: 2560,
            n_layers: 48,
            qsa_interval: 4,
            ssm_state_size: 128,
            ssm_k_heads: 16,
            ssm_v_heads: 48,
            ssm_d_conv: 4,
            ssm_conv_channels: 10240,
            ssm_value_dim: 6144,
            n_head: 24,
            n_head_kv: 2,
            head_dim: 256,
            idx_q_heads: 4,
            idx_key_dim: 128,
            hc: 4,
            hc_lr: 320,
            n_expert: 512,
            n_ff: 640,
        }
    }
}

impl ModelGeometry {
    pub fn hc_dim(&self) -> i64 {
        self.hc * self.n_embd
    }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention. Derived, not a second
    /// list.
    pub fn n_qsa_layers(&self) -> i64 {
        self.n_layers / self.qsa_interval
    }
    pub fn n_gdn_layers(&self) -> i64 {
        self.n_layers - self.n_qsa_layers()
    }
    /// True for the full-attention layers. `docs/semantics.md` gives this twice over —
    /// `full_attention_interval = 4` and an explicit `attention.compress_ratios` array — and
    /// this is the first of the two.
    pub fn is_qsa_layer(&self, layer: i64) -> bool {
        layer % self.qsa_interval == self.qsa_interval - 1
    }
}

/// What the loader had to DO for a tensor, decided by `tools/pack_index.py` and written into
/// the index rather than inferred here. Discriminants match the C++ enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum WeightKind {
    /// quantized planes, or F16 already stored as 2 bytes: copy straight through
    Verbatim = 0,
    /// the pack holds the bf16 value promoted to f32: take the HIGH 16 BITS (exact)
    Bf16InF32 = 1,
    /// copy 4 bytes per element
    F32 = 2,
    /// an f16 value promoted to f32: a real f32->f16 conversion, not a truncation
    F16InF32 = 3,
}

impl WeightKind {
    /// Bytes per element in the ENGINE form. This is what turns a kind mismatch into a real
    /// check rather than a label: a consumer that casts either form to `*const f32` reads 2x
    /// the length of a 2 B/elem tensor and walks off the end.
    fn bytes_per_element(self) -> u64 {
        match self {
            WeightKind::Bf16InF32 => 2,
            _ => 4,
        }
    }
}

/// The part of a loaded tensor a shape check reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightRef {
    pub bytes: u64,
    /// the manifest's own shape, `ne0` the CONTIGUOUS axis
    pub ne0: i64,
    pub ne1: i64,
    pub elements: i64,
    pub kind: WeightKind,
}

/// Name -> tensor, implemented by whoever holds the loaded table (the loader port, or a test
/// fixture). `find` returns `None` for a tensor the pack does not have, which is information:
/// a GDN layer has no `attn_q` and a QSA layer has no `attn_qkv`.
pub trait WeightLookup {
    fn find(&self, name: &str) -> Option<&WeightRef>;
}

/// One layer's tensors, resolved by NAME. Holds a borrow of the table, so the table must
/// outlive the view.
pub struct LayerView<'a> {
    table: &'a dyn WeightLookup,
    layer: i64,
}

impl<'a> LayerView<'a> {
    pub fn new(table: &'a dyn WeightLookup, layer: i64) -> Self {
        LayerView { table, layer }
    }
    pub fn layer(&self) -> i64 {
        self.layer
    }
    pub fn name(&self, suffix: &str) -> String {
        format!("blk.{}.{}", self.layer, suffix)
    }
    pub fn get(&self, suffix: &str) -> Option<&WeightRef> {
        self.table.find(&self.name(suffix))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Either,
    Qsa,
    Gdn,
}

impl Family {
    fn applies(self, qsa: bool) -> bool {
        match self {
            Either => true,
            Qsa => qsa,
            Gdn => !qsa,
        }
    }
}

/// A geometry field a requirement is read from, so the tables below stay declarative and the
/// values come from the geometry the caller passed rather than from literals.
#[derive(Debug, Clone, Copy)]
enum Field {
    /// the residual width
    Emb,
    HcDim,
    Hc,
    HcLr,
    NExpert,
    NFf,
    SsmConvChannels,
    SsmValueDim,
    SsmDConv,
    SsmVHeads,
    SsmStateSize,
    HeadDim,
    /// 2 * n_head * head_dim
    TwoHeadsHeadDim,
    /// n_head_kv * head_dim
    KvHeadsHeadDim,
    /// n_head * head_dim
    HeadsHeadDim,
    /// idx_q_heads * idx_key_dim
    IdxQHeadsKeyDim,
    IdxKeyDim,
}

impl Field {
    fn value(self, g: &ModelGeometry) -> i64 {
        match self {
            Field::Emb => g.n_embd,
            Field::HcDim => g.hc_dim(),
            Field::Hc => g.hc,
            Field::HcLr => g.hc_lr,
            Field::NExpert => g.n_expert,
            Field::NFf => g.n_ff,
            Field::SsmConvChannels => g.ssm_conv_channels,
            Field::SsmValueDim => g.ssm_value_dim,
            Field::SsmDConv => g.ssm_d_conv,
            Field::SsmVHeads => g.ssm_v_heads,
            Field::SsmStateSize => g.ssm_state_size,
            Field::HeadDim => g.head_dim,
            Field::TwoHeadsHeadDim => 2 * g.n_head * g.head_dim,
            Field::KvHeadsHeadDim => g.n_head_kv * g.head_dim,
            Field::HeadsHeadDim => g.n_head * g.head_dim,
            Field::IdxQHeadsKeyDim => g.idx_q_heads * g.idx_key_dim,
            Field::IdxKeyDim => g.idx_key_dim,
        }
    }
}

/// A required 2-D shape. `ne0` is the CONTIGUOUS axis, matching the manifest and `s_gemv`'s
/// convention (`y[o] = sum_i x[i]*W[i][o]`, row o contiguous of length ne0).
#[derive(Debug, Clone, Copy)]
struct Want2 {
    suffix: &'static str,
    family: Family,
    ne: [Field; 2],
}

/// A required element count and ENGINE FORM for a 1-D tensor.
///
/// `kind` is not decoration. The engine form is whatever `WeightKind` the loader applied —
/// `F32` copies 4 B/elem, `Bf16InF32` re-rounds to 2 — and a consumer that casts either one
/// to `*const f32` reads 2x its length. **That is not hypothetical:
/// `ffn_gate_inp_shexp.weight` is BF16, so the arena holds 5120 B for 2560 elements, and
/// `shared_expert` read it as f32. Every one of the 2560 MoE outputs came out non-finite** —
/// a wrong answer loud enough to notice, which is the lucky version. Reading 5120 B past the
/// end of a tensor inside a 4.5 GiB arena does not fault, so the same mistake on a tensor
/// followed by plausible bytes would have produced plausible logits.
#[derive(Debug, Clone, Copy)]
struct Want1 {
    suffix: &'static str,
    family: Family,
    elements: Field,
    kind: WeightKind,
}

/// The 2-D tensor set, per layer family.
const WANT2: &[Want2] = &[
    // gated residual, EVERY layer — `gr_read` takes w_down (hc_lr, hc_dim) and w_up
    // (hc_lr, hc_dim) after its own transpose, so the pack's orientation is
    // [hc_dim, hc_lr] and [hc_lr, hc_dim].
    Want2 {
        suffix: "hc_attn_down.weight",
        family: Either,
        ne: [Field::HcDim, Field::HcLr],
    },
    Want2 {
        suffix: "hc_attn_up.weight",
        family: Either,
        ne: [Field::HcLr, Field::HcDim],
    },
    Want2 {
        suffix: "hc_attn_inject.weight",
        family: Either,
        ne: [Field::HcDim, Field::Hc],
    },
    Want2 {
        suffix: "hc_ffn_down.weight",
        family: Either,
        ne: [Field::HcDim, Field::HcLr],
    },
    Want2 {
        suffix: "hc_ffn_up.weight",
        family: Either,
        ne: [Field::HcLr, Field::HcDim],
    },
    Want2 {
        suffix: "hc_ffn_inject.weight",
        family: Either,
        ne: [Field::HcDim, Field::Hc],
    },
    // MoE, EVERY layer
    Want2 {
        suffix: "ffn_gate_inp.weight",
        family: Either,
        ne: [Field::Emb, Field::NExpert],
    },
    Want2 {
        suffix: "ffn_gate_shexp.weight",
        family: Either,
        ne: [Field::Emb, Field::NFf],
    },
    Want2 {
        suffix: "ffn_up_shexp.weight",
        family: Either,
        ne: [Field::Emb, Field::NFf],
    },
    Want2 {
        suffix: "ffn_down_shexp.weight",
        family: Either,
        ne: [Field::NFf, Field::Emb],
    },
    // GDN only
    Want2 {
        suffix: "attn_qkv.weight",
        family: Gdn,
        ne: [Field::Emb, Field::SsmConvChannels],
    },
    Want2 {
        suffix: "attn_gate.weight",
        family: Gdn,
        ne: [Field::Emb, Field::SsmValueDim],
    },
    Want2 {
        suffix: "ssm_out.weight",
        family: Gdn,
        ne: [Field::SsmValueDim, Field::Emb],
    },
    Want2 {
        suffix: "ssm_conv1d.weight",
        family: Gdn,
        ne: [Field::SsmDConv, Field::SsmConvChannels],
    },
    Want2 {
        suffix: "ssm_alpha.weight",
        family: Gdn,
        ne: [Field::Emb, Field::SsmVHeads],
    },
    Want2 {
        suffix: "ssm_beta.weight",
        family: Gdn,
        ne: [Field::Emb, Field::SsmVHeads],
    },
    // QSA only
    Want2 {
        suffix: "attn_q.weight",
        family: Qsa,
        ne: [Field::Emb, Field::TwoHeadsHeadDim],
    },
    Want2 {
        suffix: "attn_k.weight",
        family: Qsa,
        ne: [Field::Emb, Field::KvHeadsHeadDim],
    },
    Want2 {
        suffix: "attn_v.weight",
        family: Qsa,
        ne: [Field::Emb, Field::KvHeadsHeadDim],
    },
    Want2 {
        suffix: "attn_output.weight",
        family: Qsa,
        ne: [Field::HeadsHeadDim, Field::Emb],
    },
    // THE INDEXER. `q_proj` is the QUERY count and `k_proj` is the KEY width, and they are
    // different numbers — conflating them is what made the planner's indexer term 4x too big
    // in round 194.
    Want2 {
        suffix: "indexer.q_proj.weight",
        family: Qsa,
        ne: [Field::Emb, Field::IdxQHeadsKeyDim],
    },
    Want2 {
        suffix: "indexer.k_proj.weight",
        family: Qsa,
        ne: [Field::Emb, Field::IdxKeyDim],
    },
];

/// The 1-D set. Every one of these is `F32` except the shared expert's scalar gate, which is
/// BF16 — and that single exception is the one a reader would not guess, so it is written
/// down rather than inferred from the tensor count.
const WANT1: &[Want1] = &[
    Want1 {
        suffix: "hc_attn_norm.weight",
        family: Either,
        elements: Field::HcDim,
        kind: WeightKind::F32,
    },
    Want1 {
        suffix: "hc_ffn_norm.weight",
        family: Either,
        elements: Field::HcDim,
        kind: WeightKind::F32,
    },
    Want1 {
        suffix: "ffn_gate_inp_shexp.weight",
        family: Either,
        elements: Field::Emb,
        kind: WeightKind::Bf16InF32,
    },
    Want1 {
        suffix: "ssm_a",
        family: Gdn,
        elements: Field::SsmVHeads,
        kind: WeightKind::F32,
    },
    Want1 {
        suffix: "ssm_dt.bias",
        family: Gdn,
        elements: Field::SsmVHeads,
        kind: WeightKind::F32,
    },
    Want1 {
        suffix: "ssm_norm.weight",
        family: Gdn,
        elements: Field::SsmStateSize,
        kind: WeightKind::F32,
    },
    Want1 {
        suffix: "attn_q_norm.weight",
        family: Qsa,
        elements: Field::HeadDim,
        kind: WeightKind::F32,
    },
    Want1 {
        suffix: "attn_k_norm.weight",
        family: Qsa,
        elements: Field::HeadDim,
        kind: WeightKind::F32,
    },
    Want1 {
        suffix: "indexer.q_norm.weight",
        family: Qsa,
        elements: Field::IdxKeyDim,
        kind: WeightKind::F32,
    },
    Want1 {
        suffix: "indexer.k_norm.weight",
        family: Qsa,
        elements: Field::IdxKeyDim,
        kind: WeightKind::F32,
    },
];

fn check_one(table: &dyn WeightLookup, g: &ModelGeometry, layer: i64, err: &mut String) -> bool {
    let v = LayerView::new(table, layer);
    let qsa = g.is_qsa_layer(layer);

    for w in WANT2 {
        if !w.family.applies(qsa) {
            continue;
        }
        let Some(r) = v.get(w.suffix) else {
            *err = format!("layer {layer}: missing {}", v.name(w.suffix));
            return false;
        };
        let (want_ne0, want_ne1) = (w.ne[0].value(g), w.ne[1].value(g));
        if r.ne0 != want_ne0 {
            *err = shape_error(&v, w.suffix, "ne0", r.ne0, want_ne0);
            return false;
        }
        if r.ne1 != want_ne1 {
            *err = shape_error(&v, w.suffix, "ne1", r.ne1, want_ne1);
            return false;
        }
    }

    for w in WANT1 {
        if !w.family.applies(qsa) {
            continue;
        }
        let Some(r) = v.get(w.suffix) else {
            *err = format!("layer {layer}: missing {}", v.name(w.suffix));
            return false;
        };
        let want_elements = w.elements.value(g);
        if r.elements != want_elements {
            *err = shape_error(&v, w.suffix, "elements", r.elements, want_elements);
            return false;
        }
        if r.kind != w.kind {
            // The byte count is what makes this a REAL check and not a label: a 4 B/elem
            // tensor holds `elements * 4`, a 2 B/elem one holds `elements * 2`, and a
            // consumer reading it as the wrong one walks off the end.
            let want_bytes = want_elements as u64 * w.kind.bytes_per_element();
            *err = format!(
                "layer {layer}: {} is engine form {} ({} B), the kernels read it as form {} ({want_bytes} B)",
                v.name(w.suffix),
                r.kind as i32,
                r.bytes,
                w.kind as i32,
            );
            return false;
        }
    }
    true
}

fn shape_error(v: &LayerView, suffix: &str, what: &str, got: i64, want: i64) -> String {
    format!(
        "layer {}: {} {what} is {got}, the kernels require {want}",
        v.layer(),
        v.name(suffix)
    )
}

/// Geometry a shape check cannot run without: a zero interval divides by zero and a negative
/// layer count checks nothing. C++ would take the divide-by-zero as UB; here it is an error.
fn geometry_problem(g: &ModelGeometry) -> Option<String> {
    if g.qsa_interval < 1 {
        return Some(format!(
            "qsa_interval {} is not a positive interval",
            g.qsa_interval
        ));
    }
    if g.n_layers < 0 {
        return Some(format!("n_layers {} is negative", g.n_layers));
    }
    None
}

/// Every shape the kernels depend on, asserted for ONE layer. `Err` names the tensor, what it
/// has and what is required.
///
/// Deliberately separate from `LayerView`: a caller that only wants the pointers should not
/// pay for the checks, and a caller that wants the checks should get ALL of them rather than
/// the ones its own call site happens to touch.
pub fn check_layer(table: &dyn WeightLookup, g: &ModelGeometry, layer: i64) -> Result<(), String> {
    if let Some(problem) = geometry_problem(g) {
        return Err(problem);
    }
    if layer < 0 || layer >= g.n_layers {
        return Err(format!("layer {layer} is outside 0..{}", g.n_layers - 1));
    }
    let mut err = String::new();
    if check_one(table, g, layer, &mut err) {
        Ok(())
    } else {
        Err(err)
    }
}

/// `check_layer` over every layer, plus the cross-layer counts. Returns the first failure.
///
/// The count check is ported as-is, and its reachability is stated honestly: for a modulo
/// predicate the count is arithmetically `n_layers / qsa_interval` for any non-negative
/// `n_layers`, so this can disagree with the geometry only if `is_qsa_layer` stops being the
/// modulo rule (tested: `count_check_is_implied_by_the_predicate`). The guard that actually
/// fires on a real pack is the per-layer one — a QSA tensor present on a GDN layer shows up
/// as the GDN set missing, and vice versa.
pub fn check_all(table: &dyn WeightLookup, g: &ModelGeometry) -> Result<(), String> {
    if let Some(problem) = geometry_problem(g) {
        return Err(problem);
    }
    let (mut n_qsa, mut n_gdn) = (0i64, 0i64);
    for l in 0..g.n_layers {
        let mut err = String::new();
        if !check_one(table, g, l, &mut err) {
            return Err(err);
        }
        if g.is_qsa_layer(l) {
            n_qsa += 1;
        } else {
            n_gdn += 1;
        }
    }
    // The split is derived twice over in `docs/semantics.md` — the interval and an explicit
    // ratio array — and both give 36 GDN and 12 QSA. Counting them here is the check that the
    // LAYER TYPE PREDICATE and the pack agree, which no per-tensor shape check can see.
    if n_qsa != g.n_qsa_layers() || n_gdn != g.n_gdn_layers() {
        return Err(format!(
            "layer split is {n_qsa} QSA / {n_gdn} GDN, the geometry says {} / {}",
            g.n_qsa_layers(),
            g.n_gdn_layers()
        ));
    }
    Ok(())
}
