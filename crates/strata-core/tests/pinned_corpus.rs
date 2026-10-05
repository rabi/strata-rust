//! Byte-parity replay of `pinned.cu`'s deterministic core: the pin cap, the
//! slice bounds, the read plan with its per-layer checksums, and the unbuffered
//! gate.
//!
//! `tools/pinned_corpus.cpp` compiles the real `pinned.cu` host-only (g++ with
//! `-x c++` and the stub headers; the CUDA and platform symbols are stubbed to
//! abort-if-called) and prints one `CORPUS|…` line per observation. The fixture
//! files it writes are embedded in the golden as hex, so this test rebuilds them
//! and replays every case. The two line streams must be identical.
//!
//! The timing fields are deliberately absent from the golden - they measure the
//! machine. Everything else is compared.
//!
//! Regenerate (needs a Strata checkout; build line in the harness header):
//!   ./target/corpus/pccorpus 2>/dev/null > crates/strata-core/tests/golden/pinned.txt

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use strata_core::pinned::{
    arena_pin_cap_gib, experts_unbuffered, load_experts_direct, load_experts_ranges, uniform_bounds,
};
use strata_core::{fnv, fnv_seeded};

const GOLDEN: &str = include_str!("golden/pinned.txt");

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
            "strata-pinned-replay-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&root).unwrap();
        for (rel, bytes) in files {
            std::fs::write(root.join(rel), bytes).unwrap();
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

fn replay(tree: &Path) -> Vec<String> {
    let mut out = Vec::new();

    // arena_pin_cap_gib - the env value, and what atoi makes of it
    let mut cap_case = |tag: &str, value: Option<&str>| {
        out.push(format!(
            "CORPUS|CAP|{tag}|value={}",
            arena_pin_cap_gib(value)
        ));
    };
    cap_case("unset", None);
    cap_case("empty", Some(""));
    cap_case("auto", Some("auto"));
    cap_case("five", Some("5"));
    cap_case("zero", Some("0"));
    cap_case("negative", Some("-3"));
    cap_case("not_a_number", Some("abc"));
    cap_case("leading_digits", Some("7x"));
    cap_case("leading_space", Some(" 9"));

    // uniform_bounds is not in the golden: it lives in an anonymous namespace
    // inside pinned.cu, so the harness cannot call it. The dedicated unit test
    // below pins it instead.

    // load_experts_ranges
    let mut load_case = |tag: &str,
                         file: &str,
                         off: &[u64],
                         bytes: &[u64],
                         threads: usize,
                         chunk: u64,
                         dst_size: usize| {
        let mut dst = vec![0u8; dst_size];
        let st = load_experts_ranges(&tree.join(file), &mut dst, off, bytes, threads, chunk);
        out.push(format!(
            "CORPUS|LOAD|{tag}|ok={}|bytes={}|layers={}|error={}|dst_hash={:016x}|checksums={}",
            u64::from(st.ok),
            st.bytes,
            st.layers,
            st.error,
            fnv(&dst),
            st.layer_checksums
                .iter()
                .map(|c| format!("{c},"))
                .collect::<String>()
        ));
    };
    load_case(
        "clean_1t",
        "pack_full.bin",
        &[0, 1024, 2048],
        &[1024, 1024, 1024],
        1,
        4096,
        3072,
    );
    load_case(
        "clean_4t",
        "pack_full.bin",
        &[0, 1024, 2048],
        &[1024, 1024, 1024],
        4,
        4096,
        3072,
    );
    load_case(
        "small_chunks",
        "pack_full.bin",
        &[0, 1024, 2048],
        &[1024, 1024, 1024],
        3,
        256,
        3072,
    );
    load_case(
        "ragged_chunk",
        "pack_full.bin",
        &[0, 1024, 2048],
        &[1024, 1024, 1024],
        2,
        700,
        3072,
    );
    load_case(
        "zero_layer",
        "pack_full.bin",
        &[0, 1024, 2048],
        &[1024, 0, 1024],
        2,
        512,
        3072,
    );
    load_case(
        "gap",
        "pack_gap.bin",
        &[0, 2048],
        &[1024, 1024],
        2,
        512,
        4096,
    );
    load_case(
        "short_read",
        "pack_short.bin",
        &[0, 1024, 2048],
        &[1024, 1024, 1024],
        1,
        4096,
        3072,
    );
    load_case(
        "seek_past_eof",
        "pack_short.bin",
        &[0, 2048],
        &[512, 512],
        1,
        256,
        4096,
    );
    load_case(
        "threads_zero",
        "pack_full.bin",
        &[0, 1024],
        &[1024, 1024],
        0,
        512,
        3072,
    );

    // load_experts_direct - the Windows path; Linux returns the defaults
    let mut direct_case = |tag: &str, file: &str, off: &[u64], bytes: &[u64], dst_size: usize| {
        let mut dst = vec![0u8; dst_size];
        let st = load_experts_direct(&tree.join(file), &mut dst, off, bytes);
        out.push(format!(
            "CORPUS|DIRECT|{tag}|ok={}|bytes={}|layers={}|error={}|dst_hash={:016x}",
            u64::from(st.ok),
            st.bytes,
            st.layers,
            st.error,
            fnv(&dst)
        ));
    };
    direct_case(
        "direct_clean",
        "pack_full.bin",
        &[0, 1024, 2048],
        &[1024, 1024, 1024],
        3072,
    );
    direct_case(
        "direct_unaligned",
        "pack_full.bin",
        &[0, 1024, 2048],
        &[1024, 1024, 1024],
        3072,
    );

    // experts_unbuffered
    let mut unbuf_case = |tag: &str, value: Option<&str>| {
        let (r, why) = experts_unbuffered(value);
        out.push(format!("CORPUS|UNBUF|{tag}|ret={}|why={why}", u64::from(r)));
    };
    unbuf_case("unset", None);
    unbuf_case("empty", Some(""));
    unbuf_case("one", Some("1"));
    unbuf_case("zero", Some("0"));
    unbuf_case("word", Some("auto"));

    out
}

#[test]
fn pinned_matches_cpp() {
    let tree = Tree::new(&fixtures());
    let got = replay(&tree.root);
    let want = observations();
    assert_eq!(got.len(), want.len(), "observation count");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "observation {i}");
    }
}

/// The bounds lines are the one section the harness could not call, so pin the
/// reading against the C++ source directly: a lone remainder is not a slice.
#[test]
fn bounds_tail_is_the_last_full_slice_end() {
    assert_eq!(uniform_bounds(5000, 1024), vec![0, 1024, 2048, 3072, 4096]);
    assert_eq!(uniform_bounds(4096, 1024), vec![0, 1024, 2048, 3072, 4096]);
    assert_eq!(uniform_bounds(1024, 4096), Vec::<u64>::new());
    assert_eq!(uniform_bounds(0, 1024), Vec::<u64>::new());
    assert_eq!(uniform_bounds(4096, 0), Vec::<u64>::new());
}

/// The chaining the read plan relies on: the seed is the running hash, not a
/// fresh offset each chunk.
#[test]
fn fnv_chains() {
    let a = fnv_seeded(b"hello", 0);
    assert_ne!(a, fnv_seeded(b"hello", 1));
    assert_eq!(a, fnv_seeded(b"hello", 0));
}
