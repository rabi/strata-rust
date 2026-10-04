// @generated: ported from Strata include/strata/spec/suffix_drafter.hpp + src/spec/suffix_drafter.cpp

//! The suffix-lookup drafter (no weights, no GPU).
//!
//! Proposes the tokens that followed the longest earlier occurrence of the
//! sequence's current suffix: when the output quotes its input (edits, refactors,
//! repeated code) the continuation is usually exact, and a verify pass accepts many
//! tokens at once. It needs no model and costs microseconds.
//!
//! Index: every trigram of the history (prompt + accepted output) maps to its
//! `WAYS` most recent end positions in a fixed-size open-addressing table, so
//! memory is bounded (~20 bytes per history token) and appends are O(1). A
//! proposal checks those candidates, extends each match backwards up to
//! `max_match`, and takes the longest (most recent on ties). Matches shorter than
//! `min_match` propose nothing.

const WAYS: usize = 4;

#[derive(Clone, Copy, PartialEq)]
struct Slot {
    /// trigram hash + 1 (0 = empty)
    key: u64,
    /// end positions, most recent first
    pos: [u32; WAYS],
    n: u8,
}

const EMPTY_SLOT: Slot = Slot {
    key: 0,
    pos: [0; WAYS],
    n: 0,
};

pub struct SuffixDrafter {
    min_match: usize,
    max_match: usize,
    hist: Vec<i32>,
    table: Vec<Slot>,
    mask: usize,
    last_match: usize,
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}

impl Default for SuffixDrafter {
    fn default() -> Self {
        Self::new(3, 32, 1 << 19)
    }
}

impl SuffixDrafter {
    pub const WAYS: usize = WAYS;

    pub fn new(min_match: usize, max_match: usize, capacity_tokens: usize) -> Self {
        let min_match = min_match.max(3);
        let max_match = max_match.max(min_match);
        let mut cap = 1usize;
        while cap < capacity_tokens * 2 {
            cap <<= 1; // load factor <= 0.5 at the nominal capacity
        }
        SuffixDrafter {
            min_match,
            max_match,
            hist: Vec::new(),
            table: vec![EMPTY_SLOT; cap],
            mask: cap - 1,
            last_match: 0,
        }
    }

    pub fn reset(&mut self) {
        self.hist.clear();
        self.table.iter_mut().for_each(|s| *s = EMPTY_SLOT);
        self.last_match = 0;
    }

    fn key_at(&self, end: usize) -> u64 {
        let a = self.hist[end - 2] as u32 as u64;
        let b = self.hist[end - 1] as u32 as u64;
        let c = self.hist[end] as u32 as u64;
        mix(a.wrapping_mul(0x9E3779B97F4A7C15) ^ mix(b.wrapping_add(0x632BE59BD9B4E019)) ^ (c << 1))
            | 1
    }

    fn find_slot(&mut self, key: u64, insert: bool) -> Option<usize> {
        let mut i = key as usize & self.mask;
        for _ in 0..=self.mask {
            let s = &self.table[i];
            if s.key == key {
                return Some(i);
            }
            if s.key == 0 {
                if insert {
                    self.table[i].key = key;
                    return Some(i);
                }
                return None;
            }
            i = (i + 1) & self.mask;
        }
        None // table full: the history outgrew its capacity
    }

    /// Add tokens to the history (the prompt, then every accepted token).
    pub fn append_slice(&mut self, tokens: &[i32]) {
        for &t in tokens {
            self.hist.push(t);
            let end = self.hist.len() - 1;
            if end < 2 {
                continue;
            }
            let key = self.key_at(end);
            let idx = match self.find_slot(key, true) {
                Some(i) => i,
                None => continue,
            };
            let slot = &mut self.table[idx];
            for w in (1..WAYS).rev() {
                slot.pos[w] = slot.pos[w - 1];
            }
            slot.pos[0] = end as u32;
            if slot.n < WAYS as u8 {
                slot.n += 1;
            }
        }
    }

    /// Add one token to the history.
    pub fn append(&mut self, token: i32) {
        self.append_slice(&[token]);
    }

    /// Returns up to `max_k` proposed next tokens (empty = no match of at least
    /// min_match).
    pub fn propose(&mut self, max_k: usize) -> Vec<i32> {
        self.last_match = 0;
        let n = self.hist.len();
        if n < 4 || max_k == 0 {
            return Vec::new();
        }
        let cur = n - 1;
        let key = self.key_at(cur);
        let idx = match self.find_slot(key, false) {
            Some(i) => i,
            None => return Vec::new(),
        };
        let slot = self.table[idx];
        let mut best_end = 0usize;
        let mut best_len = 0usize;
        for w in 0..slot.n as usize {
            let p = slot.pos[w] as usize;
            if p >= cur {
                continue; // the current suffix itself
            }
            let mut len = 0usize;
            while len < self.max_match && len <= p && self.hist[p - len] == self.hist[cur - len] {
                len += 1;
            }
            if len > best_len {
                best_len = len;
                best_end = p;
            } // most recent first, so ties keep the newer
        }
        if best_len < self.min_match {
            return Vec::new();
        }
        self.last_match = best_len;
        // The continuation may run into the current suffix (periodic text); reading
        // history up to `cur` is valid.
        let mut out = Vec::new();
        let mut q = best_end + 1;
        while q <= cur && out.len() < max_k {
            out.push(self.hist[q]);
            q += 1;
        }
        out
    }

    /// Length of the match behind the last proposal (0 if none).
    pub fn last_match(&self) -> usize {
        self.last_match
    }

    pub fn size(&self) -> usize {
        self.hist.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ported from Strata src/spec/suffix_drafter_test.cpp
    #[test]
    fn repeat_proposes_the_continuation() {
        let mut d = SuffixDrafter::default();
        let doc: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
        d.append_slice(&doc);
        d.append_slice(&[10, 11, 12]);
        let out = d.propose(5);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0], 13);
        assert_eq!(out[4], 17);
        assert_eq!(d.last_match(), 3);
    }

    #[test]
    fn no_match_proposes_nothing() {
        let mut d = SuffixDrafter::default();
        d.append_slice(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(d.propose(4).len(), 0);
    }

    #[test]
    fn longest_match_preferred_over_most_recent() {
        let mut d = SuffixDrafter::default();
        let doc: Vec<i32> = vec![
            7, 8, 1, 2, 3, 100, 101, 9, 1, 2, 3, 200, 201, 50, 7, 8, 1, 2, 3,
        ];
        d.append_slice(&doc);
        let out = d.propose(2);
        assert_eq!(out, vec![100, 101]);
        assert_eq!(d.last_match(), 5);
    }

    #[test]
    fn min_match_respected() {
        let mut d = SuffixDrafter::new(4, 32, 1 << 19);
        d.append_slice(&[1, 2, 3, 9, 5, 2, 3]);
        assert_eq!(d.propose(2).len(), 0);
    }

    #[test]
    fn periodic_continuation() {
        let mut d = SuffixDrafter::default();
        d.append_slice(&[1, 2, 3, 1, 2, 3, 1, 2, 3]);
        let out = d.propose(6);
        assert!(out.len() >= 3);
        assert_eq!(&out[..3], &[1, 2, 3]);
    }

    #[test]
    fn propose_after_overflow_of_nominal_capacity() {
        let mut d = SuffixDrafter::new(3, 32, 1024);
        let doc: Vec<i32> = (0..5000).map(|i| i % 997).collect();
        d.append_slice(&doc);
        assert!(!d.propose(4).is_empty());
    }

    #[test]
    fn reset_clears_history_and_matches() {
        let mut d = SuffixDrafter::default();
        d.append_slice(&[10, 11, 12, 13, 10, 11, 12]);
        assert!(!d.propose(2).is_empty());
        d.reset();
        assert_eq!(d.size(), 0);
        assert_eq!(d.propose(2).len(), 0);
        assert_eq!(d.last_match(), 0);
    }
}
