//! Byte-parity replay of `expert_source.cpp`'s CPU-only policy: the host-RAM
//! gate (MemAvailable + the cgroup v1/v2 walk), the cache-complement planner,
//! the complement/mapped resolution, the adaptive-tier swap, and the resident
//! keep decision.
//!
//! `tools/expert_plan_corpus.cpp` compiles the REAL `expert_source.cpp` on the
//! host (CUDA symbols stubbed to abort-if-called; the kernel/platform symbols
//! stay unresolved because nothing under test reaches them) and prints one
//! `CORPUS|…` line per observation. That output is checked in as
//! `golden/expert_plan.txt`; the fixture files the C++ wrote (the fake /proc and
//! cgroup tree) are embedded in it as hex, so this test rebuilds them and replays
//! every case through the Rust port. The two line streams must be identical.
//!
//! Regenerate (needs a Strata checkout; build line in the harness header):
//!   ./target/corpus/epcorpus > crates/strata-core/tests/golden/expert_plan.txt

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use strata_core::expert_plan::{
    cgroup_available_bytes, choose_resident_keep_from, exchange_cache_complement,
    host_available_memory, make_cache_complement_plan, NO_CACHE_COMPLEMENT,
};

const GOLDEN: &str = include_str!("golden/expert_plan.txt");

// ------------------------------------------------------------------ golden I/O

fn observations() -> Vec<String> {
    GOLDEN
        .lines()
        .filter(|l| l.starts_with("CORPUS|") && !l.starts_with("CORPUS|FILE|"))
        .map(String::from)
        .collect()
}

fn fixtures() -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for line in GOLDEN.lines() {
        if let Some(rest) = line.strip_prefix("CORPUS|FILE|") {
            let (rel, hex) = rest.split_once('|').expect("FILE line");
            out.insert(rel.to_string(), unhex(hex));
        }
    }
    out
}

fn unhex(hex: &str) -> Vec<u8> {
    assert!(hex.len().is_multiple_of(2), "odd hex length");
    hex.as_bytes()
        .chunks(2)
        .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap())
        .collect()
}

struct Tree {
    root: PathBuf,
}

impl Tree {
    fn new(files: &BTreeMap<String, Vec<u8>>) -> Tree {
        let root = std::env::temp_dir().join(format!(
            "strata-ep-replay-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&root).unwrap();
        for (rel, bytes) in files {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, bytes).unwrap();
        }
        Tree { root }
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

// ------------------------------------------------------------------ the matrix

/// The same cases, in the same order, as the harness.
fn replay(tree: &Path) -> Vec<String> {
    let gib = 1u64 << 30;
    let mut out = Vec::new();

    // cgroup_available_bytes
    let mut cgroup_case =
        |tag: &str, limit: u64, valid: bool, current: u64, inactive: u64, dirty: u64, wb: u64| {
            let stat = strata_core::expert_plan::CgroupMemoryStat {
                current,
                inactive_file: inactive,
                file_dirty: dirty,
                file_writeback: wb,
                valid,
            };
            match cgroup_available_bytes(limit, &stat) {
                Some(bytes) => out.push(format!("CORPUS|CGROUP|{tag}|ok=1|bytes={bytes}")),
                None => out.push(format!("CORPUS|CGROUP|{tag}|ok=0|bytes=0")),
            }
        };
    cgroup_case(
        "invalid_stat",
        100 * gib,
        false,
        50 * gib,
        20 * gib,
        gib,
        gib,
    );
    cgroup_case(
        "plain",
        100 * gib,
        true,
        50 * gib,
        20 * gib,
        5 * gib,
        2 * gib,
    );
    cgroup_case(
        "dirty_eats_reclaim",
        100 * gib,
        true,
        50 * gib,
        20 * gib,
        20 * gib,
        0,
    );
    cgroup_case(
        "writeback_eats_reclaim",
        100 * gib,
        true,
        50 * gib,
        20 * gib,
        0,
        20 * gib,
    );
    cgroup_case(
        "inactive_over_charged",
        100 * gib,
        true,
        10 * gib,
        20 * gib,
        0,
        0,
    );
    cgroup_case(
        "over_limit_after_reclaim",
        10 * gib,
        true,
        50 * gib,
        20 * gib,
        0,
        0,
    );
    cgroup_case("all_zero", 100 * gib, true, 0, 0, 0, 0);
    cgroup_case("limit_zero", 0, true, 0, 0, 0, 0);
    cgroup_case(
        "dirty_over_reclaim",
        100 * gib,
        true,
        50 * gib,
        20 * gib,
        30 * gib,
        0,
    );

    // make_cache_complement_plan
    let mut plan_case = |tag: &str,
                         n_layers: i64,
                         n_expert: i64,
                         blob_bytes: &[u64],
                         primary: &[(i32, i32)],
                         additional: &[(i32, i32)],
                         show: usize| {
        let shown =
            match make_cache_complement_plan(n_layers, n_expert, blob_bytes, primary, additional) {
                Ok((offsets, bytes)) => {
                    let sentinels = offsets
                        .iter()
                        .filter(|o| **o == NO_CACHE_COMPLEMENT)
                        .count();
                    (
                        format!(
                            "ok=1|err=|bytes={bytes}|count={}|sentinels={sentinels}",
                            offsets.len()
                        ),
                        offsets,
                        sentinels,
                    )
                }
                Err(e) => (
                    format!("ok=0|err={e}|bytes=0|count=0|sentinels=0"),
                    Vec::new(),
                    0,
                ),
            };
        let _ = shown.2;
        let mut line = format!("CORPUS|PLAN|{tag}|{}", shown.0);
        line.push_str("|first=");
        for o in shown.1.iter().take(show) {
            line.push_str(&format!("{o},"));
        }
        out.push(line);
    };
    plan_case("no_pairs", 2, 3, &[100, 200], &[], &[], 12);
    plan_case("primary_marks", 2, 3, &[100, 200], &[(0, 1)], &[], 12);
    plan_case("additional_marks", 2, 3, &[100, 200], &[], &[(1, 0)], 12);
    plan_case("both_tiers", 2, 3, &[100, 200], &[(0, 1)], &[(1, 0)], 12);
    plan_case("overlap", 2, 3, &[100, 200], &[(0, 1)], &[(0, 1)], 0);
    plan_case("dup_primary", 2, 3, &[100, 200], &[(0, 1), (0, 1)], &[], 0);
    plan_case(
        "dup_additional",
        2,
        3,
        &[100, 200],
        &[],
        &[(0, 1), (0, 1)],
        0,
    );
    plan_case("pair_outside", 2, 3, &[100, 200], &[(2, 0)], &[], 0);
    plan_case("pair_negative", 2, 3, &[100, 200], &[(0, -1)], &[], 0);
    plan_case("zero_blob", 2, 3, &[100, 0], &[], &[], 0);
    plan_case("bad_layers", 0, 3, &[100], &[], &[], 0);
    plan_case("bad_expert", 2, 0, &[100, 200], &[], &[], 0);
    plan_case("size_mismatch", 2, 3, &[100], &[], &[], 0);
    plan_case(
        "huge_geometry",
        1_000_000_000,
        1_000_000_000,
        &[1],
        &[],
        &[],
        0,
    );
    plan_case("per_layer_sizes", 3, 2, &[7, 11, 13], &[(1, 1)], &[], 12);

    // exchange_cache_complement
    let mut exchange_case = |tag: &str, mut offsets: Vec<u64>, inp: usize, out_index: usize| {
        let before = offsets.clone();
        let ok = exchange_cache_complement(&mut offsets, inp, out_index);
        let mut line = format!(
            "CORPUS|EXCHANGE|{tag}|ok={}|changed={}",
            ok as i32,
            i32::from(offsets != before)
        );
        for i in 0..offsets.len().min(8) {
            if offsets[i] != before[i] {
                line.push_str(&format!("|{i}:{}->{}", before[i], offsets[i]));
            }
        }
        out.push(line);
    };
    exchange_case("swap", vec![0, NO_CACHE_COMPLEMENT, 100], 2, 1);
    exchange_case("same_index", vec![0, 100], 1, 1);
    exchange_case("in_not_in_copy", vec![NO_CACHE_COMPLEMENT, 100], 0, 1);
    exchange_case("out_already_in", vec![0, 100], 0, 1);
    exchange_case("in_out_of_range", vec![0, 100], 5, 1);
    exchange_case("out_of_range", vec![0, 100], 0, 5);

    // cache_complement_blob_or_fallback: which backing, and the offset into it.
    let mut fallback_case = |tag: &str, offsets: &[u64], index: usize, have_complement: bool| {
        let got = strata_core::expert_plan::cache_complement_blob_or_fallback(
            index,
            offsets,
            have_complement,
        );
        out.push(match got {
            Some(off) => format!("CORPUS|FALLBACK|{tag}|which=complement|off={off}"),
            None => format!("CORPUS|FALLBACK|{tag}|which=mapped|off=0"),
        });
    };
    fallback_case("in_complement", &[0, 100, 200], 1, true);
    fallback_case("sentinel_goes_mapped", &[0, NO_CACHE_COMPLEMENT], 1, true);
    fallback_case("no_complement_backing", &[0, 100], 1, false);
    fallback_case("index_past_end", &[0, 100], 5, true);

    // choose_resident_keep_from
    let mut keep_case = |tag: &str, slot_bytes: &[u64], base: u64, budget: u64, lend_from: i64| {
        out.push(format!(
            "CORPUS|KEEP|{tag}|keep={}",
            choose_resident_keep_from(slot_bytes, base, budget, lend_from)
        ));
    };
    keep_case("all_fit", &[100, 100, 100], 0, 1000, 0);
    keep_case("base_over_budget", &[100], 2000, 1000, 0);
    keep_case("partial", &[100, 100, 100], 0, 250, 0);
    keep_case("exact_fit", &[100, 100], 0, 200, 0);
    keep_case("lend_from_clamps", &[100, 100, 100], 0, 1000, 99);
    keep_case("lend_from_negative", &[100, 100, 100], 0, 1000, -1);
    keep_case("nothing_kept", &[100, 100], 0, 1000, 2);
    keep_case("zero_budget", &[100], 0, 0, 0);

    // host_available_memory against the rebuilt tree.
    let mut host_case = |tag: &str| {
        let root = tree.join(tag);
        let m = host_available_memory(&root.join("cgroup_file"), &root.join("groups"), &root);
        match m {
            Some(m) => out.push(format!(
                "CORPUS|HOST|{tag}|ok=1|available={}|cgroup_limit={}",
                m.available, m.cgroup_limit
            )),
            None => out.push(format!(
                "CORPUS|HOST|{tag}|ok=0|available=0|cgroup_limit={}",
                u64::MAX
            )),
        }
    };
    for tag in [
        "no_cgroup_line",
        "no_memavailable",
        "groups_file_missing",
        "v2_root_no_controllers",
        "v2_max",
        "v2_limit",
        "v2_tighter_than_meminfo",
        "v2_unreadable_limit",
        "v2_missing_stat",
        "v2_duplicate_key",
        "v2_ancestor_and_child",
        "v1_memory_controller",
        "v1_unlimited",
    ] {
        host_case(tag);
    }

    out
}

#[test]
fn expert_plan_matches_cpp() {
    let tree = Tree::new(&fixtures());
    let got = replay(&tree.root);
    let want = observations();
    assert_eq!(got.len(), want.len(), "observation count");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "observation {i}");
    }
}
