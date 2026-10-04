// @generated: ported from Strata include/strata/spec/draft_policy.hpp + src/spec/draft_policy.cpp

//! Per verify round: the MTP's window, or a lookup (suffix) window?
//!
//! The policy learns both sides online and compares expected committed tokens per
//! millisecond, and takes the lookup window only when its best E/cost beats the
//! MTP's by `margin`. Costs are the measured round times per window size (EMA;
//! sizes not seen yet are scaled from seen ones by a prior shape), so the policy
//! adapts to the machine and the context length. It only chooses which drafts to
//! verify: the output is unchanged.

pub const K_SHAPE: [f64; DraftPolicy::K_MAX_T + 1] =
    [0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6];
const K_COST_ALPHA: f64 = 0.1; // EMA weight of a new round time
const K_TOK_ALPHA: f64 = 0.05; // EMA weight of a new MTP window outcome
const K_DECAY: f64 = 0.97; // lookup counts: older windows fade
                           // Before a bucket has data: the longer the match, the likelier its continuation.
                           // Worth 4 observations, so a few real windows override it.
const K_PRIOR_Q: [f64; DraftPolicy::K_BUCKETS] = [0.75, 0.88, 0.93, 0.96];
const K_PRIOR_N: f64 = 4.0;
const K_PROBES: f64 = 3.0; // a lookup window size is tried this often before its guessed cost can veto it

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pick {
    pub lookup: bool,
    /// window size (1 + drafts)
    pub t: usize,
}

pub struct DraftPolicy {
    max_t: usize,
    margin: f64,
    cost: [f64; Self::K_MAX_T + 1],
    cost_n: [f64; Self::K_MAX_T + 1],
    mtp_tok: [f64; Self::K_MAX_T + 1],
    mtp_n: [f64; Self::K_MAX_T + 1],
    ok: [f64; Self::K_BUCKETS],
    bad: [f64; Self::K_BUCKETS],
}

impl DraftPolicy {
    pub const K_MAX_T: usize = 8;
    pub const K_BUCKETS: usize = 4;

    pub fn new(max_t: usize) -> Self {
        Self::with_margin(max_t, 0.03)
    }

    pub fn with_margin(max_t: usize, margin: f64) -> Self {
        DraftPolicy {
            max_t: max_t.clamp(1, Self::K_MAX_T),
            margin,
            cost: [0.0; Self::K_MAX_T + 1],
            cost_n: [0.0; Self::K_MAX_T + 1],
            mtp_tok: [0.0; Self::K_MAX_T + 1],
            mtp_n: [0.0; Self::K_MAX_T + 1],
            ok: [0.0; Self::K_BUCKETS],
            bad: [0.0; Self::K_BUCKETS],
        }
    }

    fn bucket(match_len: i32) -> usize {
        if match_len < 6 {
            0
        } else if match_len < 12 {
            1
        } else if match_len < 24 {
            2
        } else {
            3
        }
    }

    /// Current acceptance rate q for a match length.
    pub fn lookup_rate(&self, match_len: i32) -> f64 {
        let b = Self::bucket(match_len);
        (self.ok[b] + K_PRIOR_N * K_PRIOR_Q[b]) / (self.ok[b] + self.bad[b] + K_PRIOR_N)
    }

    /// Measured or scaled round time of a window of `t` tokens.
    pub fn cost_ms(&self, t: usize) -> f64 {
        let t = t.clamp(1, Self::K_MAX_T);
        if self.cost_n[t] > 0.0 {
            return self.cost[t];
        }
        // scale from the measured sizes, weighting each by how often it was seen
        let mut num = 0.0;
        let mut den = 0.0;
        for u in (1..=Self::K_MAX_T).filter(|&u| self.cost_n[u] > 0.0) {
            let w = self.cost_n[u].min(20.0);
            num += w * self.cost[u] * K_SHAPE[t] / K_SHAPE[u];
            den += w;
        }
        if den > 0.0 {
            num / den
        } else {
            K_SHAPE[t]
        }
    }

    fn mtp_tokens(&self, t: usize) -> f64 {
        if self.mtp_n[t] > 0.0 {
            self.mtp_tok[t]
        } else {
            1.0 + 0.7 * (t as f64 - 1.0) // before any MTP window of this size: a typical acceptance
        }
    }

    /// `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none);
    /// `match_len`: its match length.
    pub fn choose(&self, t_mtp: usize, lookup_k: usize, match_len: i32) -> Pick {
        let mut p = Pick {
            lookup: false,
            t: t_mtp.clamp(1, self.max_t),
        };
        if lookup_k == 0 {
            return p;
        }
        let base = self.mtp_tokens(p.t) / self.cost_ms(p.t);
        let q = self.lookup_rate(match_len);
        let mut e = 1.0;
        let mut qi = 1.0;
        let mut best = 0.0;
        let mut best_t = 0usize;
        for k in 1..=lookup_k.min(self.max_t - 1) {
            qi *= q;
            e += qi;
            let r = e / self.cost_ms(k + 1);
            if r > best {
                best = r;
                best_t = k + 1;
            }
        }
        if best_t > 0 && best > base * (1.0 + self.margin) {
            p.lookup = true;
            p.t = best_t;
            return p;
        }
        // A guessed cost can keep the policy from ever measuring a size: the first few
        // times a confident lookup would need a size not measured yet, it is tried
        // (verification keeps the output; only the one round's speed is at stake).
        let t_full = lookup_k.min(self.max_t - 1) + 1;
        if t_full > p.t && self.cost_n[t_full] < K_PROBES && q >= 0.85 {
            p.lookup = true;
            p.t = t_full;
        }
        p
    }

    /// After the round: the window it used, the drafts accepted, and the round's time
    /// (verify + commit + draft).
    pub fn observe(
        &mut self,
        lookup: bool,
        t: usize,
        accepted: i32,
        match_len: i32,
        round_ms: f64,
    ) {
        let t = t.clamp(1, Self::K_MAX_T);
        if round_ms > 0.0 {
            self.cost[t] = if self.cost_n[t] > 0.0 {
                (1.0 - K_COST_ALPHA) * self.cost[t] + K_COST_ALPHA * round_ms
            } else {
                round_ms
            };
            self.cost_n[t] += 1.0;
        }
        if lookup {
            let b = Self::bucket(match_len);
            self.ok[b] = K_DECAY * self.ok[b] + accepted as f64;
            self.bad[b] = K_DECAY * self.bad[b] + if accepted < t as i32 - 1 { 1.0 } else { 0.0 };
        } else {
            let got = accepted as f64 + 1.0;
            self.mtp_tok[t] = if self.mtp_n[t] > 0.0 {
                (1.0 - K_TOK_ALPHA) * self.mtp_tok[t] + K_TOK_ALPHA * got
            } else {
                got
            };
            self.mtp_n[t] += 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ported from Strata src/spec/draft_policy_test.cpp
    fn cost(t: usize) -> f64 {
        19.0 + 10.5 * (t as f64 - 1.0) // ms per round, measured shape
    }

    #[test]
    fn no_proposal_keeps_the_mtp_window() {
        let mut p = DraftPolicy::new(6);
        for _ in 0..50 {
            p.observe(false, 4, 2, 0, cost(4)); // MTP windows of 4: 3 tokens each
        }
        let k = p.choose(4, 0, 0);
        assert!(!k.lookup && k.t == 4);
    }

    #[test]
    fn rejected_short_match_drafts_stop_being_taken() {
        let mut p = DraftPolicy::new(6);
        for _ in 0..50 {
            p.observe(false, 4, 2, 0, cost(4));
        }
        for t in 2..=6 {
            p.observe(false, t, 0, 0, cost(t));
        }
        for _ in 0..40 {
            p.observe(true, 6, 0, 4, cost(6)); // short matches, all rejected
        }
        assert!(p.lookup_rate(4) < 0.15);
        assert!(!p.choose(4, 5, 4).lookup);
    }

    #[test]
    fn accepted_long_matches_take_the_full_window() {
        let mut p = DraftPolicy::new(6);
        for _ in 0..50 {
            p.observe(false, 4, 2, 0, cost(4));
        }
        for t in 2..=6 {
            p.observe(false, t, 0, 0, cost(t));
        }
        for _ in 0..40 {
            p.observe(true, 6, 0, 4, cost(6));
        }
        for _ in 0..40 {
            p.observe(true, 6, 5, 30, cost(6)); // long matches, all accepted
        }
        assert!(p.lookup_rate(30) > 0.9);
        let k = p.choose(4, 5, 30);
        assert!(k.lookup && k.t == 6);
        assert!(!p.choose(4, 5, 4).lookup, "buckets are separate");
        assert!(p.choose(4, 20, 30).t <= 6, "never beyond the window cap");
    }

    #[test]
    fn mediocre_lookup_does_not_replace_a_strong_mtp_window() {
        let mut p = DraftPolicy::new(8);
        for _ in 0..50 {
            p.observe(false, 3, 2, 0, cost(3)); // a very good MTP: 3 of 3 tokens
        }
        for t in 2..=8 {
            p.observe(false, t, (t - 1) as i32, 0, cost(t));
        }
        for _ in 0..40 {
            p.observe(true, 4, 2, 8, cost(4)); // lookup at q ~ 0.67
        }
        assert!(!p.choose(3, 7, 8).lookup);
    }

    #[test]
    fn an_unmeasured_size_is_probed_for_a_confident_lookup() {
        let mut p = DraftPolicy::new(6);
        for _ in 0..50 {
            p.observe(false, 4, 3, 0, cost(4)); // a near-perfect MTP, only size 4 seen
        }
        let k = p.choose(4, 5, 40);
        assert!(k.lookup && k.t == 6);
        for _ in 0..3 {
            p.observe(true, 6, 5, 40, 3.0 * cost(6)); // it turns out very expensive
        }
        assert!(
            !p.choose(4, 5, 40).lookup,
            "after the probes, the measured cost decides"
        );
    }
}
