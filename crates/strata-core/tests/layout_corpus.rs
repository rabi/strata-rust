//! Cross-check of `strata_core::layout` against the real C++ `src/core/layout.cpp`.
//!
//! `layout.cpp` has no dedicated C++ test — it is exercised only through the load path on
//! real packs. So the oracle here is `tools/layout_corpus.cpp`: it compiles the real
//! `layout.cpp`, runs the battery below against it, asserts each case itself, and prints
//! `CORPUS <name>\t<result>` lines. Those are recorded in `golden_layout_corpus.txt`; this
//! test replays every case through the Rust port and requires the same result, byte for
//! byte, for every golden line.
//!
//!   g++ -std=c++20 -Iinclude tools/layout_corpus.cpp src/core/layout.cpp -o /tmp/layout_corpus
//!   /tmp/layout_corpus > crates/strata-core/tests/golden_layout_corpus.txt

use std::collections::HashMap;
use strata_core::layout::{
    check_all, check_layer, LayerView, ModelGeometry, WeightKind, WeightLookup, WeightRef,
};

const GOLDEN: &str = include_str!("golden_layout_corpus.txt");

struct Table(HashMap<String, WeightRef>);

impl WeightLookup for Table {
    fn find(&self, name: &str) -> Option<&WeightRef> {
        self.0.get(name)
    }
}

fn w2(ne0: i64, ne1: i64) -> WeightRef {
    WeightRef {
        bytes: ne0 as u64 * ne1 as u64 * 2,
        ne0,
        ne1,
        elements: 0,
        kind: WeightKind::Verbatim,
    }
}

fn w1(elements: i64, kind: WeightKind) -> WeightRef {
    WeightRef {
        bytes: elements as u64 * kind_bytes(kind),
        ne0: elements,
        ne1: 1,
        elements,
        kind,
    }
}

fn kind_bytes(k: WeightKind) -> u64 {
    if k == WeightKind::Bf16InF32 {
        2
    } else {
        4
    }
}

/// A pack with every tensor the checks require, at the geometry's own dimensions — the
/// mirror of `pack_everything` in the harness.
fn pack_everything(g: &ModelGeometry) -> Table {
    let mut t = Table(HashMap::new());
    for l in 0..g.n_layers {
        let p = format!("blk.{l}.");
        let mut put = |suffix: &str, r: WeightRef| {
            t.0.insert(p.clone() + suffix, r);
        };
        put("hc_attn_down.weight", w2(g.hc_dim(), g.hc_lr));
        put("hc_attn_up.weight", w2(g.hc_lr, g.hc_dim()));
        put("hc_attn_inject.weight", w2(g.hc_dim(), g.hc));
        put("hc_ffn_down.weight", w2(g.hc_dim(), g.hc_lr));
        put("hc_ffn_up.weight", w2(g.hc_lr, g.hc_dim()));
        put("hc_ffn_inject.weight", w2(g.hc_dim(), g.hc));
        put("ffn_gate_inp.weight", w2(g.n_embd, g.n_expert));
        put("ffn_gate_shexp.weight", w2(g.n_embd, g.n_ff));
        put("ffn_up_shexp.weight", w2(g.n_embd, g.n_ff));
        put("ffn_down_shexp.weight", w2(g.n_ff, g.n_embd));
        put("hc_attn_norm.weight", w1(g.hc_dim(), WeightKind::F32));
        put("hc_ffn_norm.weight", w1(g.hc_dim(), WeightKind::F32));
        put(
            "ffn_gate_inp_shexp.weight",
            w1(g.n_embd, WeightKind::Bf16InF32),
        );
        if g.is_qsa_layer(l) {
            put("attn_q.weight", w2(g.n_embd, 2 * g.n_head * g.head_dim));
            put("attn_k.weight", w2(g.n_embd, g.n_head_kv * g.head_dim));
            put("attn_v.weight", w2(g.n_embd, g.n_head_kv * g.head_dim));
            put("attn_output.weight", w2(g.n_head * g.head_dim, g.n_embd));
            put(
                "indexer.q_proj.weight",
                w2(g.n_embd, g.idx_q_heads * g.idx_key_dim),
            );
            put("indexer.k_proj.weight", w2(g.n_embd, g.idx_key_dim));
            put("attn_q_norm.weight", w1(g.head_dim, WeightKind::F32));
            put("attn_k_norm.weight", w1(g.head_dim, WeightKind::F32));
            put("indexer.q_norm.weight", w1(g.idx_key_dim, WeightKind::F32));
            put("indexer.k_norm.weight", w1(g.idx_key_dim, WeightKind::F32));
        } else {
            put("attn_qkv.weight", w2(g.n_embd, g.ssm_conv_channels));
            put("attn_gate.weight", w2(g.n_embd, g.ssm_value_dim));
            put("ssm_out.weight", w2(g.ssm_value_dim, g.n_embd));
            put("ssm_conv1d.weight", w2(g.ssm_d_conv, g.ssm_conv_channels));
            put("ssm_alpha.weight", w2(g.n_embd, g.ssm_v_heads));
            put("ssm_beta.weight", w2(g.n_embd, g.ssm_v_heads));
            put("ssm_a", w1(g.ssm_v_heads, WeightKind::F32));
            put("ssm_dt.bias", w1(g.ssm_v_heads, WeightKind::F32));
            put("ssm_norm.weight", w1(g.ssm_state_size, WeightKind::F32));
        }
    }
    t
}

fn ok_layer(t: &dyn WeightLookup, g: &ModelGeometry, l: i64) -> String {
    check_layer(t, g, l)
        .map(|()| "ok".to_string())
        .unwrap_or_else(|e| e)
}
fn ok_all(t: &dyn WeightLookup, g: &ModelGeometry) -> String {
    check_all(t, g)
        .map(|()| "ok".to_string())
        .unwrap_or_else(|e| e)
}

/// Rebuilds every corpus case and returns name -> Rust result, in golden order.
fn rust_corpus() -> Vec<(String, String)> {
    let g = ModelGeometry::default();
    let mut cases: Vec<(String, String)> = Vec::new();
    let mut push = |name: &str, result: String| cases.push((name.to_string(), result));

    let mut t = pack_everything(&g);
    push("good_all", ok_all(&t, &g));
    push("good_layer0", ok_layer(&t, &g, 0));
    push("good_layer3", ok_layer(&t, &g, 3));

    let key = "blk.3.indexer.k_proj.weight";
    let save = t.0.remove(key).unwrap();
    push("missing_kproj", ok_layer(&t, &g, 3));
    push("missing_kproj_all", ok_all(&t, &g));
    t.0.insert(key.into(), save);

    let save = std::mem::replace(
        t.0.get_mut(key).unwrap(),
        w2(g.n_embd, g.idx_q_heads * g.idx_key_dim),
    );
    push("kproj_query_sized", ok_layer(&t, &g, 3));
    t.0.insert(key.into(), save);

    let key = "blk.0.hc_attn_up.weight";
    let save = std::mem::replace(t.0.get_mut(key).unwrap(), w2(g.hc_dim(), g.hc_lr));
    push("up_transposed", ok_layer(&t, &g, 0));
    t.0.insert(key.into(), save);

    let key = "blk.0.ffn_gate_inp.weight";
    let save = std::mem::replace(t.0.get_mut(key).unwrap(), w2(g.n_embd, 511));
    push("ne1_mismatch", ok_layer(&t, &g, 0));
    t.0.insert(key.into(), save);

    let key = "blk.0.ffn_gate_inp_shexp.weight";
    let save = std::mem::replace(t.0.get_mut(key).unwrap(), w1(g.n_embd, WeightKind::F32));
    push("shexp_gate_f32", ok_layer(&t, &g, 0));
    t.0.insert(key.into(), save);

    let key = "blk.0.ssm_norm.weight";
    let save = std::mem::replace(t.0.get_mut(key).unwrap(), w1(127, WeightKind::F32));
    push("ssm_norm_short", ok_layer(&t, &g, 0));
    t.0.insert(key.into(), save);

    push("layer_neg1", ok_layer(&t, &g, -1));
    push("layer_48", ok_layer(&t, &g, 48));

    let g5 = ModelGeometry {
        qsa_interval: 5,
        n_layers: 25,
        ..g
    };
    let t5 = pack_everything(&g5);
    push("interval5_all", ok_all(&t5, &g5));

    let t48 = pack_everything(&g);
    let g47 = ModelGeometry { n_layers: 47, ..g };
    push("truncated_47", ok_all(&t48, &g47));
    cases
}

#[test]
fn every_golden_cpp_line_matches_the_rust_port() {
    let rust: HashMap<String, String> = rust_corpus().into_iter().collect();
    let mut n = 0;
    for line in GOLDEN.lines() {
        let Some(rest) = line.strip_prefix("CORPUS ") else {
            continue; // the harness's own assertion lines
        };
        let (name, want) = rest.split_once('\t').expect("CORPUS <name>\t<result>");
        assert!(
            rust.contains_key(name),
            "corpus case '{name}' not replayed by Rust"
        );
        assert_eq!(rust[name], want, "case '{name}' diverged from the C++");
        n += 1;
    }
    assert_eq!(n, 14, "every golden case replayed");
    assert_eq!(
        rust.len(),
        n,
        "Rust replays exactly the golden cases, no more"
    );
}

#[test]
fn harness_assertions_hold_in_rust_too() {
    let g = ModelGeometry::default();
    let t = pack_everything(&g);
    // "a wrong GDN tensor on a QSA layer is not checked there" — the family filter is
    // per-layer, so a wrong tensor of the other family passes on a QSA layer.
    let mut t2 = pack_everything(&g);
    t2.0.insert("blk.3.attn_qkv.weight".into(), w2(g.n_embd, 999));
    assert_eq!(check_layer(&t2, &g, 3), Ok(()));
    // and the last layer is in range
    assert_eq!(check_layer(&t, &g, 47), Ok(()));
}

#[test]
fn the_count_check_is_implied_by_the_predicate() {
    // Honest statement of reachability: for the modulo predicate the QSA count over
    // 0..n_layers is arithmetically n_layers / qsa_interval, so check_all's cross-layer
    // count check can never fire while is_qsa_layer is the modulo rule. It guards against a
    // future in which the predicate and the derivation diverge — not against a pack. The
    // pack-level guard is the per-layer missing/shape check.
    for n in [0i64, 1, 3, 4, 5, 12, 47, 48, 49] {
        for interval in 1..=6i64 {
            let g = ModelGeometry {
                n_layers: n,
                qsa_interval: interval,
                ..Default::default()
            };
            let counted = (0..n).filter(|l| g.is_qsa_layer(*l)).count() as i64;
            assert_eq!(counted, g.n_qsa_layers(), "n={n} interval={interval}");
        }
    }
}

#[test]
fn geometry_that_cannot_be_checked_is_an_error_not_ub() {
    // C++ divides by qsa_interval unguarded (UB at interval 0); Rust refuses.
    let g_zero = ModelGeometry {
        qsa_interval: 0,
        ..Default::default()
    };
    let t = pack_everything(&ModelGeometry::default());
    assert!(check_layer(&t, &g_zero, 0)
        .unwrap_err()
        .contains("qsa_interval"));
    assert!(check_all(&t, &g_zero).unwrap_err().contains("qsa_interval"));
    let g_neg = ModelGeometry {
        n_layers: -1,
        ..Default::default()
    };
    assert!(check_all(&t, &g_neg).unwrap_err().contains("negative"));
}

#[test]
fn layer_view_names_come_from_the_layer_they_check() {
    let t = pack_everything(&ModelGeometry::default());
    let v = LayerView::new(&t, 7);
    assert_eq!(v.name("attn_q.weight"), "blk.7.attn_q.weight");
    assert!(v.get("attn_q.weight").is_some());
    assert!(
        v.get("attn_qkv.weight").is_none(),
        "layer 7 is QSA, it has no qkv"
    );
}
