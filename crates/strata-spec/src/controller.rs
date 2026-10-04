// @generated: ported from Strata include/strata/spec/controller.hpp + src/spec/controller.cpp

//! Which drafter, and how many draft tokens, this step.
//!
//! Each step the controller maximizes E[tokens committed] / T(step) over the choices
//! none (k = 0) | suffix lookup with k <= its proposal | MTP with k <= `K_MAX`,
//! using running estimates of per-position acceptance and a cost model of a
//! (k+1)-token verify pass. It keeps k = 0 unless the best choice beats plain
//! decoding by `min_gain`.

pub const K_MAX: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    None,
    Lookup,
    Mtp,
}

#[derive(Clone, Copy, Debug)]
pub struct Choice {
    pub source: Source,
    pub k: usize,
    pub expected_tokens: f64,
    pub tokens_per_ms: f64,
}

impl Default for Choice {
    fn default() -> Self {
        Choice {
            source: Source::None,
            k: 0,
            expected_tokens: 1.0,
            tokens_per_ms: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CostModel {
    /// one-token dense pass
    pub dense_ms: f64,
    /// dense cost ratio by n = k+1 tokens (index n-1)
    pub dense_ratio: [f64; K_MAX + 1],
    /// all experts on the CPU
    pub cpu_all_miss_ms: f64,
    /// share of distinct experts served from VRAM
    pub hit_rate: f64,
    /// distinct experts per window, in units of one token's worth (index n-1)
    pub distinct_ratio: [f64; K_MAX + 1],
    /// each extra token on an expert, fraction of a read
    pub extra_use_cost: f64,
    pub sync_ms: f64,
    /// per MTP draft token
    pub mtp_draft_ms: f64,
    /// per lookup draft token
    pub lookup_draft_ms: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        CostModel {
            dense_ms: 11.0,
            dense_ratio: [1.0, 1.05, 1.3, 1.45, 1.85, 2.2, 2.6, 3.0, 3.3],
            cpu_all_miss_ms: 15.8,
            hit_rate: 0.55,
            distinct_ratio: [1.0, 1.70, 2.31, 2.88, 3.40, 3.89, 4.35, 4.80, 5.2],
            extra_use_cost: 0.25,
            sync_ms: 2.4,
            mtp_draft_ms: 1.2,
            lookup_draft_ms: 0.01,
        }
    }
}

impl CostModel {
    /// Step time for verifying n = k + 1 tokens, plus drafting k tokens with the given source.
    pub fn step_ms(&self, k: usize, mtp: bool) -> f64 {
        let n = (k + 1).min(K_MAX + 1); // n = k+1 >= 1, so no lower clamp needed
        let dense = self.dense_ms * self.dense_ratio[n - 1];
        let u = self.distinct_ratio[n - 1];
        let uses_per_expert = n as f64 / u;
        let cpu = (1.0 - self.hit_rate)
            * self.cpu_all_miss_ms
            * u
            * (1.0 + self.extra_use_cost * (uses_per_expert - 1.0));
        let draft = k as f64
            * if mtp {
                self.mtp_draft_ms
            } else {
                self.lookup_draft_ms
            };
        dense + cpu + self.sync_ms + draft
    }
}

pub struct Controller {
    cost: CostModel,
    min_gain: f64,
    ema: f64,
    mtp_p: [f64; K_MAX],
    lookup_q: [f64; 4],
}

fn bucket(match_len: i32) -> usize {
    if match_len >= 16 {
        3
    } else if match_len >= 8 {
        2
    } else if match_len >= 5 {
        1
    } else {
        0
    }
}

impl Default for Controller {
    fn default() -> Self {
        Self::new(CostModel::default(), 0.05, 0.05)
    }
}

impl Controller {
    /// Priors: MTP per-position acceptance from the flyweight measurement on this
    /// model (0.86 per draft); lookup by match length from the offline replay.
    /// Both are then learned per session.
    pub fn new(cost: CostModel, min_gain: f64, ema: f64) -> Self {
        Controller {
            cost,
            min_gain,
            ema,
            mtp_p: [0.86; K_MAX],
            lookup_q: [0.35, 0.6, 0.8, 0.92],
        }
    }

    /// Expected tokens committed by a window whose drafts have per-position conditional
    /// acceptance `p` (the verify pass always commits one token). `same` reuses p[0] for
    /// every position (lookup's single rate per match bucket).
    fn expected(p: &[f64], k: usize, same: bool) -> f64 {
        let mut e = 1.0;
        let mut run = 1.0;
        for i in 0..k {
            run *= if same { p[0] } else { p[i] };
            e += run;
        }
        e
    }

    /// `lookup_available` tokens the suffix drafter can propose now, from a match of
    /// `lookup_match` tokens.
    pub fn choose(&self, lookup_available: usize, lookup_match: i32, mtp_ready: bool) -> Choice {
        let mut best = Choice {
            source: Source::None,
            k: 0,
            expected_tokens: 1.0,
            tokens_per_ms: 1.0 / self.cost.step_ms(0, false),
        };
        let baseline = best.tokens_per_ms;
        let consider = |best: &mut Choice, s: Source, k: usize, e: f64| {
            let rate = e / self.cost.step_ms(k, s == Source::Mtp);
            if rate > best.tokens_per_ms {
                *best = Choice {
                    source: s,
                    k,
                    expected_tokens: e,
                    tokens_per_ms: rate,
                };
            }
        };
        let lk = lookup_available.min(K_MAX);
        let q = self.lookup_q[bucket(lookup_match)];
        for k in 1..=lk {
            consider(
                &mut best,
                Source::Lookup,
                k,
                Self::expected(&[q; K_MAX], k, true),
            );
        }
        if mtp_ready {
            for k in 1..=K_MAX {
                consider(
                    &mut best,
                    Source::Mtp,
                    k,
                    Self::expected(&self.mtp_p, k, false),
                );
            }
        }
        if best.source != Source::None && best.tokens_per_ms < baseline * (1.0 + self.min_gain) {
            return Choice {
                tokens_per_ms: baseline,
                ..Choice::default()
            };
        }
        best
    }

    /// After verification: `accepted` of the `k` drafted tokens matched (a prefix).
    pub fn observe(&mut self, c: &Choice, accepted: usize, lookup_match: i32) {
        if c.source == Source::None || c.k == 0 {
            return;
        }
        let accepted = accepted.min(c.k);
        // Positions 0..accepted-1 were accepted given their prefix; position `accepted`
        // (if drafted) was rejected.
        let seen = c.k.min(accepted + 1);
        for i in 0..seen {
            let hit = if i < accepted { 1.0 } else { 0.0 };
            match c.source {
                Source::Mtp => self.mtp_p[i] += self.ema * (hit - self.mtp_p[i]),
                Source::Lookup => {
                    let b = bucket(lookup_match);
                    self.lookup_q[b] += self.ema * (hit - self.lookup_q[b]);
                }
                Source::None => {}
            }
        }
        // Everything drafted was accepted: the positions beyond the window were never
        // tested, so without this they keep their prior forever and the window can never
        // grow. Pull them toward the deepest observed rate.
        if c.source == Source::Mtp && accepted == c.k && c.k < K_MAX {
            for i in c.k..K_MAX {
                self.mtp_p[i] += self.ema * (self.mtp_p[c.k - 1] - self.mtp_p[i]);
            }
        }
    }

    /// Conditional acceptance of MTP draft `position` (0 = first draft token).
    pub fn mtp_accept(&self, position: usize) -> f64 {
        self.mtp_p[position]
    }

    /// Acceptance rate learned for matches of `match_len`'s bucket.
    pub fn lookup_accept(&self, match_len: i32) -> f64 {
        self.lookup_q[bucket(match_len)]
    }

    pub fn cost(&self) -> &CostModel {
        &self.cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ported from Strata src/spec/controller_test.cpp
    #[test]
    fn mtp_chosen_with_k_in_range_at_the_prior() {
        let c = Controller::default();
        let ch = c.choose(0, 0, true);
        assert!(
            ch.source == Source::Mtp && (2..=5).contains(&ch.k),
            "got {:?} k={}",
            ch.source,
            ch.k
        );
    }

    #[test]
    fn persistent_rejection_turns_speculation_off() {
        let mut c = Controller::default();
        let mut ch = c.choose(0, 0, true);
        for _ in 0..400 {
            c.observe(&ch, 0, 0);
            ch = c.choose(0, 0, true);
            if ch.source == Source::None {
                break;
            }
        }
        assert_eq!(ch.source, Source::None);
    }

    #[test]
    fn long_lookup_match_preferred_over_mtp() {
        let c = Controller::default();
        let ch = c.choose(8, 20, true);
        assert_eq!(ch.source, Source::Lookup);
    }

    #[test]
    fn cheap_wide_verify_takes_the_whole_lookup_window() {
        let cheap = CostModel {
            dense_ratio: [1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16],
            hit_rate: 0.95,
            ..CostModel::default()
        };
        let c = Controller::new(cheap, 0.05, 0.05);
        let ch = c.choose(8, 20, true);
        assert!(ch.source == Source::Lookup && ch.k == 8);
    }

    #[test]
    fn short_lookup_match_alone_is_not_used() {
        let c = Controller::default();
        assert_eq!(c.choose(8, 3, false).source, Source::None);
    }

    #[test]
    fn full_acceptance_is_learned_and_the_window_widens() {
        let mut c = Controller::default();
        let mut ch = c.choose(0, 0, true);
        let k0 = ch.k;
        for _ in 0..400 {
            c.observe(&ch, ch.k, 0);
            ch = c.choose(0, 0, true);
        }
        assert!(ch.k > k0 && c.mtp_accept(0) > 0.99, "k {} -> {}", k0, ch.k);
    }
}
