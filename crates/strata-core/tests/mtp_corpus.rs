//! Replay `MtpDrafter::load`'s deterministic path against the golden the C++ harness produced.
//!
//! Four size helpers (`native_q8_1_bytes`, `moe_hit_grouped_scratch_bytes`, `coupled_draft_scratch_bytes`,
//! and the ones `layer` re-exports) are transcribed on both sides — see `tools/mtp_corpus.cpp`.  A misread of
//! one of those shows up here as a match, not a mismatch.

use strata_core::layout::ModelGeometry;
use strata_core::mtp;

const IDX_OK: &str = concat!(
    "fc_embedding.weight q8_0 0 0 0 1024\n",
    "fc_hidden.weight q8_0 0 1024 1024 1024\n",
    "self_attn.q_proj.weight q8_0 0 2048 2048 1024\n",
    "self_attn.k_proj.weight q8_0 0 3072 3072 1024\n",
    "self_attn.v_proj.weight q8_0 0 4096 4096 1024\n",
    "self_attn.o_proj.weight q8_0 0 5120 5120 1024\n",
    "mlp.shared_expert.gate_proj.weight q8_0 0 6144 6144 1024\n",
    "mlp.shared_expert.up_proj.weight q8_0 0 7168 7168 1024\n",
    "mlp.shared_expert.down_proj.weight q8_0 0 8192 8192 1024\n",
    "norm.weight f32 0 9216 9216 1024\n",
    "head_bf16.weight bf16 0 10240 10240 1024\n",
);
const IDX_MALFORMED: &str = "norm.weight f32 0 0 0 1024\nshort line\n";
const IDX_MISSING_REQUIRED: &str =
    "norm.weight f32 0 0 0 1024\nnorm2.weight f32 0 1024 1024 1024\n";

/// (index, dense.bin len, experts.bin len, draft_vocab.bin len) — `None` is "the file is not there".
fn fixture(name: &str) -> (Option<&'static str>, Option<u64>, Option<u64>, Option<u64>) {
    match name {
        "ok" => (Some(IDX_OK), Some(11264), Some(0), None),
        "ok_vocab" => (Some(IDX_OK), Some(11264), Some(0), Some(256)),
        "no_idx" => (None, Some(11264), Some(0), None),
        "malformed" => (Some(IDX_MALFORMED), Some(11264), Some(0), None),
        "missing_required" => (Some(IDX_MISSING_REQUIRED), Some(2048), Some(0), None),
        "no_experts" => (Some(IDX_OK), Some(11264), None, None),
        _ => unreachable!(),
    }
}

fn geom() -> ModelGeometry {
    ModelGeometry {
        n_expert: 0,
        ..ModelGeometry::default()
    }
}

/// (tag, fixture, max_t, window, resident, fail_malloc_at, coupled)
type Scenario = (&'static str, &'static str, i32, i64, i64, usize, bool);

const SCENARIOS: &[Scenario] = &[
    ("t4_res0_window0", "ok", 4, 0, 0, 0, false),
    ("t4_res0_window32", "ok", 4, 32, 0, 0, false),
    ("t4_res4096_window32", "ok", 4, 32, 4096, 0, false),
    ("t4_res4096_window0", "ok", 4, 0, 4096, 0, false),
    ("t4_res4096_window0_off", "ok", 4, 0, 4096, 0, false),
    ("t4_res100000_window32", "ok", 4, 32, 100000, 0, false),
    ("t4_res100000_window0", "ok", 4, 0, 100000, 0, false),
    ("t4_res100000_window4095", "ok", 4, 4095, 100000, 0, false),
    ("t4_res100000_window5000", "ok", 4, 5000, 100000, 0, false),
    ("t1_res100000_window32", "ok", 1, 32, 100000, 0, false),
    ("t8_res100000_window32", "ok", 8, 32, 100000, 0, false),
    ("t0", "ok", 0, 32, 100000, 0, false),
    ("t9", "ok", 9, 32, 100000, 0, false),
    ("no_idx", "no_idx", 4, 32, 100000, 0, false),
    ("malformed", "malformed", 4, 32, 100000, 0, false),
    (
        "missing_required",
        "missing_required",
        4,
        32,
        100000,
        0,
        false,
    ),
    ("no_experts", "no_experts", 4, 32, 100000, 0, false),
    ("malloc_fail_dense", "ok", 4, 32, 100000, 1, false),
    ("malloc_fail_state", "ok", 4, 32, 100000, 3, false),
    ("malloc_fail_arena", "ok", 4, 32, 100000, 4, false),
    ("bind_coupled_off", "ok", 4, 0, 100000, 0, false),
    ("bind_vocab", "ok_vocab", 4, 0, 100000, 0, false),
    ("bind_coupled_on", "ok", 4, 0, 100000, 0, true),
    ("bind_vocab_coupled", "ok_vocab", 4, 0, 100000, 0, true),
];

#[test]
fn mtp_corpus_replay() {
    let g = geom();
    let mut out: Vec<String> = Vec::new();
    for (tag, fx, max_t, window, resident, fail, coupled) in SCENARIOS {
        let (index, dense, experts, vocab) = fixture(fx);
        let inp = mtp::Inputs {
            index,
            dense_bin_len: dense,
            experts_bin_len: experts,
            draft_vocab_len: vocab,
            max_t: *max_t,
            window: *window,
            max_cells: 4096,
            k: 10,
            head_row_bytes: 1024,
            n_vocab: 151936,
            coupled: *coupled,
            resident: *resident,
            fail_malloc_at: *fail,
        };
        let p = mtp::load(&g, &inp, &format!("<ROOT>/{fx}"));
        match &p.err {
            None => out.push(format!(
                "LOAD|{tag}|ok=1|err=|vram={}|kv_mode={}",
                p.vram, p.kv_mode
            )),
            Some(e) => out.push(format!(
                "LOAD|{tag}|ok=0|err={e}|vram={}|kv_mode={}",
                p.vram, p.kv_mode
            )),
        }
        for (i, (off, bytes)) in p.allocs.iter().enumerate() {
            out.push(format!("  ALLOC|{i}|off={off}|bytes={bytes}"));
        }
        if p.err.is_none() {
            let f = |n: &str| {
                p.fields
                    .iter()
                    .find(|(k, _)| *k == n)
                    .unwrap_or_else(|| panic!("no field {n} in the carve"))
                    .1
                    .unwrap()
            };
            out.push(format!(
                "  FIELDS|tok={}|ident={}|Rin={}|ident_last={}|attn_scratch={}|sh_scratch={}|grp_counts={}|probs={}",
                f("tok"),
                f("ident"),
                f("Rin"),
                p.cap - 1,
                f("attn_scratch"),
                f("sh_scratch"),
                f("grp_counts"),
                f("probs")
            ));
            out.push(format!("  BIND|head_row=1024|nv=151936|bytes={}", p.bind));
        }
    }

    let golden = include_str!("golden/mtp.txt");
    let want: Vec<&str> = golden.lines().collect();
    assert_eq!(out.len(), want.len(), "line count");
    let mut bad = 0;
    for (i, (got, exp)) in out.iter().zip(want.iter()).enumerate() {
        if got != exp {
            println!("line {i}\n  rust: {got}\n  cpp : {exp}");
            bad += 1;
            if bad > 20 {
                break;
            }
        }
    }
    assert_eq!(bad, 0, "golden mismatch");
}
