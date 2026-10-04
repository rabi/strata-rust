// @generated: ported from Strata include/strata/core/coupled_draft.hpp + src/core/coupled_draft_test.cpp

//! Coupled draft sampling's host-side arithmetic (STRATA_SPEC_COUPLED=1).
//!
//! Under sampling the verify window samples row `t` of a window at position
//! `pos0` as Philox(seed, pos0 + t), and a draft is kept only when it EQUALS that
//! row's pick. The output is therefore a function of the seed alone, whatever the
//! drafts are: drafts only decide how many tokens a window yields. The drafter
//! samples with the target's own chain and the SAME uniform the target will draw
//! for the row that verifies the draft.
//!
//! The counter arithmetic: after a window at `p` with `a` drafts accepted, the
//! next window starts at `p' = p + a + 1`. Chain step `j` runs at cell `p + a + j`
//! and predicts the token at `p + a + j + 2 = p' + j + 1`: row `j + 1` of the next
//! window, verified by row `j`'s pick, counter `p' + j`. So a draft made at cell
//! `c` is verified with counter `c + 1` - whatever the window size, the accepted
//! count, an intervening suffix-drafter window, or `draft_first` (cell `p - 1`).

use crate::sampler::penalty_rows;

/// The longest penalty window the coupled drafter mirrors (serve's kPenaltyWindowCap).
pub const K_COUPLED_HIST_CAP: usize = 4096;

/// The MTP cell of chain step j after a window at `p` with `a` drafts accepted.
pub fn coupled_draft_cell(p: i64, a: usize, j: usize) -> i64 {
    p + a as i64 + j as i64
}

/// The Philox counter the verify window will draw for the row that checks the
/// draft made at MTP cell `cell`.
pub fn coupled_draft_counter(cell: i64) -> u64 {
    (cell + 1) as u64
}

/// The penalty window the draft layer and the target use:
/// min(penalty_last_n, cap), 0 when penalties are off.
pub fn coupled_hist_len(penalty_last_n: i32, cap: usize) -> usize {
    if penalty_last_n <= 0 {
        0
    } else {
        (penalty_last_n as usize).min(cap)
    }
}

/// Where draft j's penalty window starts in the ring
/// (base in [cap - h, cap), draft i at cap + i).
pub fn coupled_hist_start(cap: usize, j: usize, h: usize) -> isize {
    cap as isize + j as isize - h as isize
}

/// The ring's base on the host: the last `h` tokens of (tail, next), -1 padded in
/// front, into `out[0..h)` - row 0 of `penalty_rows` for the one-token window
/// [next].
pub fn coupled_hist_base(tail: &[i32], next: i32, h: usize, out: &mut [i32]) {
    if h == 0 {
        return;
    }
    out[..h].copy_from_slice(&penalty_rows(tail, &[next], h)[..h]);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ported from Strata src/core/coupled_draft_test.cpp
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*: deterministic stand-in for std::mt19937(20260930)
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    // verify.cpp run(): sp.counter = pos0, and the kernel draws row t with sp.counter + t
    fn verify_row_counter(pos0: i64, t: usize) -> u64 {
        (pos0 + t as i64) as u64
    }

    /// The counters and positions of a chain of `n` drafts made after a window at
    /// `p` with `a` accepted (draft()), or for draft_first (cell0 = p - 1, a = 0),
    /// checked against the next window at `p_next`.
    fn check_chain(p: i64, a: usize, n: usize, p_next: i64) {
        for j in 0..n {
            let cell = coupled_draft_cell(p, a, j);
            // mtp: cell c predicts the token at c + 2; the next window's row j+1
            // holds position p_next + j + 1
            assert_eq!(
                cell + 2,
                p_next + j as i64 + 1,
                "draft position = its row in the next window"
            );
            // row j + 1 is kept iff it equals row j's pick, drawn at p_next + j
            assert_eq!(
                coupled_draft_counter(cell),
                verify_row_counter(p_next, j),
                "draft counter = verifying row's counter"
            );
        }
    }

    /// The device ring for one chain: base staged at [cap - h, cap) by the host,
    /// draft i at cap + i; draft j's window is [coupled_hist_start(cap, j, h), +h).
    /// Compared with penalty_rows over (consumed, [next, drafts...]).
    fn check_history(rng: &mut Rng, cap: usize, h: usize, n_consumed: usize, n_drafts: usize) {
        let consumed: Vec<i32> = (0..n_consumed).map(|_| rng.below(1000) as i32).collect();
        let drafts: Vec<i32> = (0..n_drafts).map(|_| rng.below(1000) as i32).collect();
        let next = rng.below(1000) as i32;
        let mut mapped = vec![777777i32; cap];
        let mut ring = vec![888888i32; cap + n_drafts];
        // set_draft_history stages the base at [cap - h, cap)
        let mut base = vec![0i32; h];
        coupled_hist_base(&consumed, next, h, &mut base);
        mapped[cap - h..cap].copy_from_slice(&base);
        // coupled_stage_kernel copies it into the ring
        ring[cap - h..cap].copy_from_slice(&mapped[cap - h..cap]);
        // the next window: [next, drafts...], T = n_drafts + 1 rows; row j checks draft j
        let mut window = vec![next];
        window.extend_from_slice(&drafts);
        let rows = penalty_rows(&consumed, &window, h);
        for j in 0..n_drafts {
            let s = coupled_hist_start(cap, j, h);
            assert!(
                s >= 0 && (s as usize) + h <= cap + n_drafts,
                "the window lies inside the ring"
            );
            let s = s as usize;
            for i in 0..h {
                assert_eq!(
                    ring[s + i],
                    rows[j * h + i],
                    "draft history = verify row history (cap {cap} h {h} j {j} i {i})"
                );
            }
            // coupled_merge_kernel appends the draft after sampling it
            ring[cap + j] = drafts[j];
        }
    }

    #[test]
    fn serve_style_decode_keeps_counters_and_rows_aligned() {
        let mut rng = Rng(20260930);
        for run in 0..200 {
            let mut p = 100 + run * 37; // n - 1
            let mut t_size = 1usize;
            for _ in 0..300 {
                let a = rng.below(t_size); // uniform 0..T-1
                let p_next = p + a as i64 + 1;
                check_chain(p, a, 7, p_next); // the chain makes up to max_t - 1 drafts
                p = p_next;
                t_size = 1 + rng.below(8); // the draft policy chooses the next window's size
            }
        }
    }

    #[test]
    fn draft_first_uses_cell_p_minus_one() {
        // the CLI's draft_first: the first window at p, drafts from cell p - 1 with a = 0
        let mut p = 1i64;
        while p < 5000 {
            check_chain(p - 1, 0, 7, p);
            p += 13;
        }
    }

    #[test]
    fn the_penalty_ring_matches_penalty_rows() {
        let mut rng = Rng(20260930);
        for &cap in &[8usize, 64, K_COUPLED_HIST_CAP] {
            for &h in &[1usize, 3, 8] {
                if h > cap {
                    continue;
                }
                for &n_consumed in &[0usize, 1, 2, 5, 7, 20, 100] {
                    for &n_drafts in &[1usize, 3, 7] {
                        check_history(&mut rng, cap, h, n_consumed, n_drafts);
                    }
                }
            }
            check_history(&mut rng, cap, cap, 3, 7); // the window is the whole ring base
            check_history(&mut rng, cap, cap, 5000, 7);
        }
    }

    #[test]
    fn penalty_window_length_matches_the_targets() {
        // serve caps the history at 4096: hist_n = min(max(pln,0), 4096), target = min(pln, hist_n)
        for &pln in &[-1i32, 0, 1, 64, 4095, 4096, 4097, 100000] {
            let hist_n = pln.clamp(0, 4096) as usize;
            let target = if hist_n > 0 {
                (pln as usize).min(hist_n)
            } else {
                0
            };
            assert_eq!(
                coupled_hist_len(pln, K_COUPLED_HIST_CAP),
                target,
                "penalty_last_n {pln}"
            );
        }
    }
}
