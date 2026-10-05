//! Port of `include/strata/core/conversation_buffer.hpp` — snapshot-owned host
//! storage, and the arithmetic the RAM admission decision is made against.
//!
//! A parked conversation's K/V lives here. Appending never relocates an existing
//! payload (only the small segment directory can move), so a capture that is
//! already half-written can never invalidate a pointer a transfer is using.
//! That ownership rule is also why `allocation_peak` has to be exact rather than
//! "roughly the payload size": `conversation_snapshot_capture_bytes` admits a
//! capture against THIS function, and an estimate under the real allocation
//! overcommits RAM — the failure mode the sizing PRs kept hitting.
//!
//! Every number here is checked byte-for-byte against the C++ in
//! `tests/conversation_corpus.rs` (`CORPUS|BUFFER` lines): `size`, `bytes`,
//! `allocation_peak(n)` and a fingerprint of the payload, over fresh captures at
//! segment boundaries, twelve doublings, shrinks, and a pop across a boundary.

/// C++ `sizeof(std::vector<uint8_t>)` on libstdc++/x86-64 — three pointers. The
/// directory cost in [`ConversationBuffer::bytes`] is counted in these units, so
/// the two representations must agree on the size, not only on the contents.
const SEGMENT_DIRECTORY_ENTRY: usize = 24;
const _: () = assert!(std::mem::size_of::<Vec<u8>>() == SEGMENT_DIRECTORY_ENTRY);

/// Snapshot-owned host storage: a list of byte segments that never move once
/// allocated.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConversationBuffer {
    segments: Vec<Vec<u8>>,
    size: usize,
}

impl ConversationBuffer {
    /// A segment never exceeds this; small chat turns extend the last segment
    /// instead of adding one transfer per turn.
    pub const SEGMENT_BYTES: usize = 16 * 1024 * 1024;

    pub fn new() -> Self {
        Self::default()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Bytes actually held: every segment's CAPACITY plus the directory's.
    /// Capacity, not size, because freed tail space is still resident.
    pub fn bytes(&self) -> usize {
        self.segments.capacity() * SEGMENT_DIRECTORY_ENTRY
            + self.segments.iter().map(|s| s.capacity()).sum::<usize>()
    }

    /// Peak bytes held while resizing to `n`, including the old directory a
    /// replacement is allocated next to. [`usize::MAX`] on overflow: saturation
    /// makes an overflowing estimate FAIL admission instead of passing it.
    pub fn allocation_peak(&self, n: usize) -> usize {
        self.allocation_peak_checked(n).unwrap_or(usize::MAX)
    }

    fn allocation_peak_checked(&self, n: usize) -> Option<usize> {
        let mut total = self.bytes();
        if n <= self.size {
            return Some(total);
        }
        let spare = match self.segments.last() {
            Some(last) => last.capacity() - last.len(),
            None => 0,
        };
        let extra = n - self.size - (n - self.size).min(spare);
        let full = extra / Self::SEGMENT_BYTES;
        let tail = extra % Self::SEGMENT_BYTES;
        total = total.checked_add(full.saturating_mul(Self::SEGMENT_BYTES))?;
        if tail != 0 {
            // The C++ passes `segment_bytes` as `previous` when whole segments were
            // added, so the ramp is already at its bound by then.
            let previous = if full != 0 {
                Self::SEGMENT_BYTES
            } else {
                self.segments.last().map_or(0, |s| s.capacity())
            };
            let cap = Self::segment_capacity(tail, previous, self.size != 0);
            total = total.checked_add(cap)?;
        }
        let mut count = self.segments.len();
        count = count.checked_add(full + usize::from(tail != 0))?;
        if count > self.segments.capacity() {
            let dir = count.checked_mul(SEGMENT_DIRECTORY_ENTRY)?;
            total = total.checked_add(dir)?;
        }
        Some(total)
    }

    /// The C++ `segment_capacity`: a fresh capture allocates exactly its payload;
    /// later appends grow geometrically, bounded by `SEGMENT_BYTES`.
    fn segment_capacity(count: usize, previous: usize, growing: bool) -> usize {
        if !growing {
            return count;
        }
        let ramp = 65536usize
            .max(previous.min(Self::SEGMENT_BYTES / 2) * 2)
            .min(Self::SEGMENT_BYTES);
        count.max(ramp)
    }

    /// Grow or shrink to `n`, filling new bytes with `value`. Existing payload
    /// bytes are never moved or rewritten.
    pub fn resize(&mut self, n: usize, value: u8) {
        if n > self.size {
            // `growing` is sampled once, before any segment is pushed: the first
            // segment of a capture allocates exactly its payload even though the
            // same call later appends more.
            let growing = self.size != 0;
            let spare = match self.segments.last() {
                Some(last) => last.capacity() - last.len(),
                None => 0,
            };
            let extend = (n - self.size).min(spare);
            let extra = n - self.size - extend;
            // libstdc++ `reserve` allocates exactly what it is told; so does
            // `reserve_exact`, and `bytes()` counts directory capacity.
            self.segments.reserve_exact(
                extra / Self::SEGMENT_BYTES
                    + usize::from(!extra.is_multiple_of(Self::SEGMENT_BYTES)),
            );
            if extend != 0 {
                if let Some(last) = self.segments.last_mut() {
                    last.resize(last.len() + extend, value);
                }
                self.size += extend;
            }
            while self.size < n {
                let count = (n - self.size).min(Self::SEGMENT_BYTES);
                let previous = self.segments.last().map_or(0, |s| s.capacity());
                let cap = Self::segment_capacity(count, previous, growing);
                let mut segment = Vec::with_capacity(cap);
                segment.resize(count, value);
                self.segments.push(segment);
                self.size += count;
            }
        } else {
            while let Some(last_len) = self.segments.last().map(|s| s.len()) {
                if self.size - last_len < n {
                    break;
                }
                self.segments.pop();
                self.size -= last_len;
            }
            if self.size > n {
                let trim = self.size - n;
                if let Some(last) = self.segments.last_mut() {
                    last.truncate(last.len() - trim);
                }
            }
            self.size = n;
        }
    }

    /// Drop the last byte. A C++ `pop_back` on an empty buffer underflows its
    /// size and tries to allocate; here it is a no-op.
    pub fn pop_back(&mut self) {
        self.resize(self.size.saturating_sub(1), 0);
    }

    /// `f(segment_slice, absolute_offset)` over the byte runs covering
    /// `[offset, offset + count)`, in order; the first `false` aborts. Out-of-range
    /// is a `false` return, not a panic: callers probe extents with it.
    pub fn visit<F: FnMut(&mut [u8], usize) -> bool>(
        &mut self,
        offset: usize,
        count: usize,
        mut f: F,
    ) -> bool {
        if offset > self.size || count > self.size - offset {
            return false;
        }
        let (mut at, mut left) = (offset, count);
        let mut begin = 0usize;
        for segment in self.segments.iter_mut() {
            if left == 0 {
                break;
            }
            let end = begin + segment.len();
            if at < end {
                let n = left.min(end - at);
                let from = at - begin;
                if !f(&mut segment[from..from + n], at) {
                    return false;
                }
                at += n;
                left -= n;
            }
            begin = end;
        }
        left == 0
    }

    /// [`Self::visit`] for read-only walks (restore, verify).
    pub fn visit_const<F: FnMut(&[u8], usize) -> bool>(
        &self,
        offset: usize,
        count: usize,
        mut f: F,
    ) -> bool {
        if offset > self.size || count > self.size - offset {
            return false;
        }
        let (mut at, mut left) = (offset, count);
        let mut begin = 0usize;
        for segment in self.segments.iter() {
            if left == 0 {
                break;
            }
            let end = begin + segment.len();
            if at < end {
                let n = left.min(end - at);
                let from = at - begin;
                if !f(&segment[from..from + n], at) {
                    return false;
                }
                at += n;
                left -= n;
            }
            begin = end;
        }
        left == 0
    }

    /// Copy `[offset, offset + count)` out. The C++ signature takes a raw
    /// destination; the slice length is what the range has to be.
    pub fn read(&self, target: &mut [u8], offset: usize, count: usize) -> bool {
        if target.len() < count {
            return false;
        }
        self.visit_const(offset, count, |bytes, at| {
            target[at - offset..at - offset + bytes.len()].copy_from_slice(bytes);
            true
        })
    }

    /// Content equality, ignoring how the bytes happen to be segmented.
    pub fn same_contents(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        let (mut ai, mut bi) = (0usize, 0usize);
        let (mut ax, mut bx) = (0usize, 0usize);
        while ai < self.segments.len() && bi < other.segments.len() {
            let left = &self.segments[ai];
            let right = &other.segments[bi];
            let n = (left.len() - ax).min(right.len() - bx);
            if left[ax..ax + n] != right[bx..bx + n] {
                return false;
            }
            ax += n;
            bx += n;
            if ax == left.len() {
                ai += 1;
                ax = 0;
            }
            if bx == right.len() {
                bi += 1;
                bx = 0;
            }
        }
        true
    }
}

/// Overflow-checked accumulation, from `conversation_checked.hpp`. Saturation is
/// the point: an estimate that overflows must fail admission, not wrap into a
/// number that fits.
pub(crate) fn add(total: &mut usize, n: usize) -> bool {
    match total.checked_add(n) {
        Some(v) => {
            *total = v;
            true
        }
        None => false,
    }
}

/// Overflow-checked product of factors, from `conversation_checked.hpp`. A factor
/// that does not fit in `usize` is refused before it can be multiplied.
pub(crate) fn product(out: &mut usize, factors: &[u64]) -> bool {
    let mut value: usize = 1;
    for &factor in factors {
        let Ok(f) = usize::try_from(factor) else {
            return false;
        };
        if f != 0 && value > usize::MAX / f {
            return false;
        }
        value *= f;
    }
    *out = value;
    true
}
