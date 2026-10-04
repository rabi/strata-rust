//! Numerical parity with the C++ implementation.
//!
//! `src/spec/controller_test.cpp` prints its decisions and a human reads them; here
//! the same scenarios are asserted numerically, so a Rust port that drifts fails a
//! test rather than a review. Every expected value below was copied from the C++
//! binary's output on this machine (build: g++ -std=c++20 -Iinclude
//! src/spec/controller_test.cpp src/spec/controller.cpp).

use super::controller::{Choice, Controller, CostModel, Source, K_MAX};

fn name(s: Source) -> &'static str {
    match s {
        Source::None => "none",
        Source::Mtp => "mtp",
        Source::Lookup => "lookup",
    }
}

/// (p, source, k, 1000*tokens_per_ms) — the printed sweep from controller_test.cpp.
const SWEEP: [(f64, &str, usize, f64); 7] = [
    (0.50, "none", 0, 48.8),
    (0.60, "mtp", 1, 56.1),
    (0.70, "mtp", 1, 58.7),
    (0.80, "mtp", 1, 63.9),
    (0.86, "mtp", 3, 70.8),
    (0.90, "mtp", 3, 76.3),
    (0.95, "mtp", 5, 84.6),
];

/// The acceptance stream from controller_test.cpp. The LCG is deliberately the C++
/// one — `r = 2654435761 * (i+1)` then `r = r*1103515245 + 12345`, accepted while
/// `(r >> 8) % 1000 < p*1000` — because the point is that both implementations see
/// the *same* stream and must land on the same decision.
fn sweep(p: f64) -> Choice {
    let mut c = Controller::default();
    let mut ch = c.choose(0, 0, true);
    for i in 0..2000usize {
        let mut acc = 0usize;
        let mut r = 2654435761u32.wrapping_mul(i as u32 + 1);
        while acc < ch.k && {
            r = r.wrapping_mul(1103515245).wrapping_add(12345);
            ((r >> 8) % 1000) as f64
        } < p * 1000.0
        {
            acc += 1;
        }
        c.observe(&ch, acc, 0);
        ch = c.choose(0, 0, true);
    }
    ch
}

#[test]
fn acceptance_sweep_matches_the_cpp_controller() {
    for (p, src, k, tps) in SWEEP {
        let ch = sweep(p);
        assert_eq!((name(ch.source), ch.k), (src, k), "at acceptance p={p}");
        assert!(
            (1000.0 * ch.tokens_per_ms - tps).abs() < 0.05,
            "at p={p}: {:.1} tok/s, C++ prints {tps}",
            1000.0 * ch.tokens_per_ms
        );
    }
}

/// The four printed scenarios, with the numbers the C++ test prints.
#[test]
fn printed_scenarios_match() {
    // "prior: mtp k=3 expected 3.24 tokens, 72.8 tok/s (plain 48.8)"
    let c = Controller::default();
    let ch = c.choose(0, 0, true);
    assert_eq!((name(ch.source), ch.k), ("mtp", 3));
    assert!(
        (ch.expected_tokens - 3.24).abs() < 0.005
            && (1000.0 * ch.tokens_per_ms - 72.8).abs() < 0.05,
        "prior: {} tokens, {:.1} tok/s",
        ch.expected_tokens,
        1000.0 * ch.tokens_per_ms
    );
    let plain = 1000.0 / c.cost().step_ms(0, false);
    assert!(
        (plain - 48.8).abs() < 0.05,
        "plain throughput: {plain:.1}, C++ prints 48.8"
    );

    // "after rejections: none, p0 0.42"
    let mut c = Controller::default();
    let mut ch = c.choose(0, 0, true);
    for _ in 0..400 {
        c.observe(&ch, 0, 0);
        ch = c.choose(0, 0, true);
        if ch.source == Source::None {
            break;
        }
    }
    assert_eq!(name(ch.source), "none");
    assert!(
        (c.mtp_accept(0) - 0.42).abs() < 0.005,
        "p0 after the rejection loop: {:.3}, C++ prints 0.42",
        c.mtp_accept(0)
    );

    // "long lookup match, measured costs: lookup k=3 expected 3.55, 86.8 tok/s"
    let c = Controller::default();
    let l = c.choose(8, 20, true);
    assert_eq!((name(l.source), l.k), ("lookup", 3));
    assert!(
        (l.expected_tokens - 3.55).abs() < 0.005 && (1000.0 * l.tokens_per_ms - 86.8).abs() < 0.05,
        "lookup: {} tokens, {:.1} tok/s",
        l.expected_tokens,
        1000.0 * l.tokens_per_ms
    );

    // "long lookup match, cheap wide verify: lookup k=8 expected 6.60, 328.3 tok/s"
    let cheap = CostModel {
        dense_ratio: [1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16],
        hit_rate: 0.95,
        ..CostModel::default()
    };
    let c2 = Controller::new(cheap, 0.05, 0.05);
    let ch2 = c2.choose(8, 20, true);
    assert_eq!((name(ch2.source), ch2.k), ("lookup", K_MAX));
    assert!(
        (ch2.expected_tokens - 6.60).abs() < 0.005
            && (1000.0 * ch2.tokens_per_ms - 328.3).abs() < 0.05,
        "cheap verify: {} tokens, {:.1} tok/s",
        ch2.expected_tokens,
        1000.0 * ch2.tokens_per_ms
    );

    // "after full acceptance: mtp k=8 (was 3), p0 1.000"
    let mut c = Controller::default();
    let mut ch = c.choose(0, 0, true);
    let k0 = ch.k;
    for _ in 0..400 {
        c.observe(&ch, ch.k, 0);
        ch = c.choose(0, 0, true);
    }
    assert_eq!((name(ch.source), ch.k), ("mtp", K_MAX));
    assert_eq!(k0, 3);
    assert!(c.mtp_accept(0) > 0.99, "p0 = {:.3}", c.mtp_accept(0));
}
