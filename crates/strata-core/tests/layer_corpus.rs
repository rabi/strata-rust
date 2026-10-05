//! Replay `layer.cpp`'s deterministic surface against the golden the C++ harness produced.
//!
//! The golden is `tools/layer_corpus.cpp` output on the real code.  Four size helpers are transcribed on both
//! sides (see the harness header) — a misread of one of those shows up here as a match, not a mismatch.

use strata_core::layer::*;
use strata_core::layout::ModelGeometry;
use strata_core::weights::WeightRef;

/// The fake arena base the harness hands the `*_init` functions.
const BASE: i64 = 0x100000;
/// What the harness prints for a plane pointer the C++ left at null.
const NULL_OFF: u64 = (0i64 - BASE) as u64;

fn small_geom() -> ModelGeometry {
    ModelGeometry {
        n_embd: 250,
        ssm_conv_channels: 1022,
        ssm_value_dim: 6142,
        n_head: 3,
        n_head_kv: 1,
        head_dim: 64,
        idx_q_heads: 2,
        idx_key_dim: 62,
        hc: 2,
        hc_lr: 16,
        n_expert: 8,
        n_ff: 62,
        ..ModelGeometry::default()
    }
}

fn base_ref() -> WeightRef {
    WeightRef {
        code_bits: 4,
        code_bias: -16,
        group_elems: 32,
        codebook_iq4nl: true,
        has_offset: true,
        act_kind: 1,
        codes_bytes: 1024,
        scales_bytes: 256,
        offset_bytes: 128,
        bytes: 1408,
        ..Default::default()
    }
}

/// The harness prints the struct as the C++ left it, so on refusal the fields are the C++ defaults.
fn sform_line(tag: &str, r: &WeightRef) -> String {
    match sform_of(r, "t") {
        Ok(f) => format!(
            "SFORM|{tag}|ok=1|bits={}|bias={}|group={}|codebook={}|has_off={}|act={}|err=",
            f.code_bits,
            f.code_bias,
            f.group_elems,
            f.codebook as u8,
            f.has_offset as u8,
            f.act_kind
        ),
        Err(e) => {
            format!("SFORM|{tag}|ok=0|bits=2|bias=-1|group=64|codebook=0|has_off=0|act=0|err={e}")
        }
    }
}

fn planes_line(tag: &str, r: &WeightRef) -> String {
    match plane_ptrs(r, "t") {
        Ok(p) => format!(
            "PLANES|{tag}|ok=1|codes={}|scales={}|offset={}|err=",
            p.codes,
            p.scales,
            p.offset.map(|x| x as i64).unwrap_or(-1)
        ),
        Err(e) => format!("PLANES|{tag}|ok=0|codes={NULL_OFF}|scales={NULL_OFF}|offset=-1|err={e}"),
    }
}

fn layout_line(section: &str, tag: &str, bytes: u64, init: u64) -> String {
    format!("LAYOUT|{section}|{tag}|bytes={bytes}|init={init}")
}

fn off(layout: &[(&'static str, u64)], name: &str) -> u64 {
    layout
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no field {name}"))
        .1
}

fn kvplan_line(tag: &str, s: &QsaShapes, max_cells: i64, ring: i64, f: &KvFlags) -> String {
    let p = kv_plan(s, max_cells, ring, f);
    format!(
        "KVPLAN|{tag}|mode={}|pages={}|slots={}|pooled={}",
        p.mode, p.pages, p.slots, p.pooled_rows
    )
}

#[test]
fn layer_corpus_replay() {
    let mut out: Vec<String> = Vec::new();
    let g = ModelGeometry::default();
    let s = qsa_shapes(&g);
    let small = small_geom();

    out.push(format!("Q8K|0|{}", q8k_bytes(0)));
    for n in [256i64, 512, 255, 1024, 2560, 6144, 128] {
        out.push(format!("Q8K|{n}|{}", q8k_bytes(n)));
    }
    for n in [0u64, 1, 15, 16, 17, 4095] {
        out.push(format!("ALIGN16|{n}|{}", align_up16(n)));
    }

    let ok = base_ref();
    out.push(sform_line("iq4xs", &ok));
    out.push(planes_line("iq4xs", &ok));
    out.push(sform_line(
        "q2_0",
        &WeightRef {
            code_bits: 2,
            codebook_iq4nl: false,
            ..ok.clone()
        },
    ));
    out.push(planes_line(
        "q2_0",
        &WeightRef {
            code_bits: 2,
            codebook_iq4nl: false,
            ..ok.clone()
        },
    ));
    let q8 = WeightRef {
        code_bits: 8,
        has_offset: false,
        offset_bytes: 0,
        bytes: ok.codes_bytes + ok.scales_bytes,
        ..ok.clone()
    };
    out.push(sform_line("q8_0", &q8));
    out.push(planes_line("q8_0", &q8));
    out.push(sform_line(
        "f32",
        &WeightRef {
            code_bits: 0,
            ..ok.clone()
        },
    ));
    out.push(planes_line(
        "bits3",
        &WeightRef {
            code_bits: 3,
            ..ok.clone()
        },
    ));
    out.push(planes_line(
        "sum_off_by_one",
        &WeightRef {
            bytes: 1407,
            ..ok.clone()
        },
    ));
    out.push(planes_line(
        "zero_codes",
        &WeightRef {
            codes_bytes: 0,
            bytes: ok.scales_bytes + ok.offset_bytes,
            ..ok.clone()
        },
    ));
    out.push(planes_line(
        "has_offset_false",
        &WeightRef {
            has_offset: false,
            ..ok.clone()
        },
    ));
    out.push(planes_line(
        "has_offset_true_zero",
        &WeightRef {
            has_offset: true,
            offset_bytes: 0,
            bytes: ok.codes_bytes + ok.scales_bytes,
            ..ok
        },
    ));

    for (tag, gg) in [("real", &g), ("small", &small)] {
        out.push(layout_line(
            "gdn",
            tag,
            gdn_buffers_bytes(gg),
            gdn_buffers_init(gg).1,
        ));
        out.push(layout_line(
            "moe",
            tag,
            moe_buffers_bytes(gg, 8),
            moe_buffers_init(gg, 8).1,
        ));
        out.push(layout_line(
            "qsa",
            tag,
            qsa_buffers_bytes(gg, 4096),
            qsa_buffers_init(gg, 4096).1,
        ));
        out.push(layout_line(
            "block",
            tag,
            block_buffers_bytes(gg),
            block_buffers_init(gg).1,
        ));
        out.push(format!("DUMP_STRIDE|{tag}|{}", dump_stride_floats(gg)));
    }

    // field by field, real geometry
    let gdn = gdn_buffers_init(&g).0;
    for n in [
        "x_q8k",
        "x_q8_0",
        "x_bf16",
        "qkv",
        "alpha",
        "o",
        "y_q8k",
        "state",
        "conv_state",
    ] {
        out.push(format!("gdn.{n}|{}", off(&gdn, n)));
    }
    let moe = moe_buffers_init(&g, 8).0;
    for n in [
        "x_bf16",
        "logits",
        "ids",
        "shared",
        "sh_scratch",
        "x_q8_0",
        "x_q8k",
    ] {
        out.push(format!("moe.{n}|{}", off(&moe, n)));
    }
    let qsa = qsa_buffers_init(&g, 4096).0;
    for n in [
        "x_q8k",
        "x_bf16",
        "q_full",
        "idx_raw",
        "cell_scores",
        "ids",
        "k_scratch",
        "attn",
        "attn16",
        "attn_q8k",
        "attn_scratch",
    ] {
        out.push(format!("qsa.{n}|{}", off(&qsa, n)));
    }
    let block = block_buffers_init(&g).0;
    for (print, lookup) in [
        ("R", "R"),
        ("mixed", "mixed"),
        ("inject", "inject"),
        ("inject2", "inject2"),
        ("gr_rs", "gr_rs"),
        ("head_q8k", "head_q8k"),
        ("gr.xn", "xn"),
        ("gr.xq", "xq"),
        ("gr.lq", "lq"),
        ("gr.gated", "gated"),
        ("gr.lo", "lo"),
    ] {
        out.push(format!("block.{print}|{}", off(&block, lookup)));
    }
    out.push(format!(
        "GR_BYTES|{}",
        gr_workspace_init(
            &GrShapes {
                n_embd: g.n_embd,
                hc: g.hc,
                hc_lr: g.hc_lr
            },
            true
        )
        .0
    ));

    for n_ff in [640i64, 62, 0, 31] {
        out.push(format!(
            "SH_EXP|{n_ff}|{}",
            shared_expert_scratch_bytes(n_ff)
        ));
    }
    for cap in [qsa_selection_width(K_TOPK_MAX_CELLS, &s), 64, 1] {
        out.push(format!(
            "QSA_DECODE_SCRATCH|{cap}|{}",
            qsa_decode_attn_scratch_floats(cap, &s)
        ));
    }
    for fmt in [0i32, 1, 2] {
        out.push(format!("KV_BLOCK|{fmt}|{}", kv_block_bytes(&s, fmt)));
    }
    for slots in [1024i64, 1] {
        out.push(format!("KV_MAP|{slots}|{}", kv_stream_map_bytes(slots)));
    }
    out.push(format!("QSA_STEP|{}", qsa_step_bytes()));
    out.push(format!(
        "SEL_WIDTH|{}",
        qsa_selection_width(K_TOPK_MAX_CELLS, &s)
    ));

    let f0 = KvFlags::default();
    let rows: Vec<(&str, i64, i64, i64, bool, bool)> = vec![
        ("resident0_ring-1", 4096, -1, 0, false, false),
        ("resident0_ring1024", 4096, 1024, 0, false, false),
        ("resident4096_ring-1", 4096, -1, 4096, false, false),
        ("resident4096_ring0", 4096, 0, 4096, false, false),
        ("resident4096_ring0_min", 4096, 0, 100, false, false),
        ("resident4096_ring0_off", 4096, 0, 4096, false, true),
        ("resident4096_ring1024", 4096, 1024, 4096, false, false),
        ("resident4096_ring1024_off", 4096, 1024, 4096, true, false),
        ("resident4096_ring4096", 4096, 4096, 4096, false, false),
        ("resident4096_ring4092", 4096, 4092, 4096, false, false),
        ("big_resident4096_ring-1", 100000, -1, 4096, false, false),
        ("big_resident4096_ring-1_off", 100000, -1, 4096, false, true),
        ("big_resident100_ring-1", 100000, -1, 100, false, false),
        (
            "big_resident20480_ring1024",
            100000,
            1024,
            20480,
            false,
            false,
        ),
        (
            "big_resident20480_ring1024_off",
            100000,
            1024,
            20480,
            true,
            false,
        ),
        ("big_resident4096_ring0", 100000, 0, 4096, false, false),
        ("big_resident4096_ring0_off", 100000, 0, 4096, false, true),
        ("big_resident100_ring0", 100000, 0, 100, false, false),
    ];
    for (tag, max_cells, ring, resident, ring_off, main_off) in rows {
        let f = KvFlags {
            resident,
            ring_off,
            main_off,
            ..f0
        };
        out.push(kvplan_line(tag, &s, max_cells, ring, &f));
    }

    for mode in 0..3i32 {
        let mut f = KvFlags {
            int8: mode == 1,
            q4: mode == 2,
            ..f0
        };
        out.push(format!(
            "KVPOOL|mode{mode}|plain={}",
            kv_pool_bytes(&s, 1024, false, false, &f)
        ));
        out.push(format!(
            "KVPOOL|mode{mode}|int8arg={}",
            kv_pool_bytes(&s, 1024, false, true, &f)
        ));
        out.push(format!(
            "KVPOOL|mode{mode}|hybrid={}",
            kv_pool_bytes(&s, 1024, true, false, &f)
        ));
        f.hybrid = true;
        out.push(format!(
            "KVPOOL|mode{mode}|hybridflag={}",
            kv_pool_bytes(&s, 1024, true, true, &f)
        ));
        for ring in [-1i64, 1024] {
            let f2 = KvFlags {
                resident: 4096,
                ..f
            };
            out.push(format!(
                "QSA_STATE|mode{mode}|ring{ring}|rope0={}|rope1={}",
                qsa_state_bytes(&g, 4096, false, ring, &f2),
                qsa_state_bytes(&g, 4096, true, ring, &f2)
            ));
        }
    }

    let golden = include_str!("golden/layer.txt");
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
