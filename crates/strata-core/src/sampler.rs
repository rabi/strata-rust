// @generated: ported from Strata include/strata/kernels/sampler.hpp (penalty_rows)

//! Host-side penalty histories for a verify window.
//!
//! Only the pure host helpers live here. `sample_tokens` and the coupled-draft
//! sample/stage calls are CUDA kernel launches and stay in C++ behind the device
//! ABI (`strata-device`).

/// Row `t` of the result (`T` rows of `h` slots) is the last `h` tokens of
/// `tail` followed by `window[0..=t]`, most recent LAST, `-1` in the unused
/// front slots.
///
/// `tail` is what the state consumed before the window, `window[0]` the fed-back
/// token and `window[1..]` the drafts: row `t` is exactly the history plain decode
/// counts when it picks the token after `window[t]`, so drafting cannot change
/// which tokens are penalised. Row 0 alone is the single row the engine staged
/// before 0.1.19.
pub fn penalty_rows(tail: &[i32], window: &[i32], h: usize) -> Vec<i32> {
    let n_tail = tail.len() as i64;
    let mut out = vec![0i32; window.len() * h];
    for (t, row) in out.chunks_exact_mut(h).enumerate() {
        let avail = n_tail + t as i64 + 1; // tail + window[0..=t]
        let take = h.min(avail.max(0) as usize);
        row[..h - take].fill(-1);
        for j in 0..take {
            let i = avail - take as i64 + j as i64; // index into (tail, window)
            row[h - take + j] = if i < n_tail {
                tail[i as usize]
            } else {
                window[(i - n_tail) as usize]
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_zero_counts_tail_plus_the_fed_back_token() {
        let tail = [1, 2, 3, 4, 5];
        let rows = penalty_rows(&tail, &[9], 3);
        assert_eq!(rows, vec![4, 5, 9]);
    }

    #[test]
    fn short_history_is_minus_one_padded_in_front() {
        let rows = penalty_rows(&[7], &[42], 4);
        assert_eq!(rows, vec![-1, -1, 7, 42]); // avail = tail(7) + window[0](42) = 2 of 4 slots
    }

    #[test]
    fn each_row_follows_the_previous_drafts() {
        let tail = [1, 2, 3];
        let window = [8, 9, 10];
        let h = 3;
        let rows = penalty_rows(&tail, &window, h);
        assert_eq!(&rows[0..3], [2, 3, 8]);
        assert_eq!(&rows[3..6], [3, 8, 9]);
        assert_eq!(&rows[6..9], [8, 9, 10]);
    }

    #[test]
    fn h_longer_than_the_available_history_pads() {
        let rows = penalty_rows(&[1], &[2, 3], 5);
        assert_eq!(&rows[0..5], [-1, -1, -1, 1, 2]);
        assert_eq!(&rows[5..10], [-1, -1, 1, 2, 3]);
    }

    #[test]
    fn empty_tail() {
        let rows = penalty_rows(&[], &[5, 6], 2);
        assert_eq!(&rows[0..2], [-1, 5]);
        assert_eq!(&rows[2..4], [5, 6]);
    }
}
