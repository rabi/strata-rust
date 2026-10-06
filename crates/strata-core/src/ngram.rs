// @generated: ported from Strata src/ngram/ple_reader.cpp - the deterministic
// core of it. The io_thread worker loop is the part left behind: it is the
// scheduling a scripted clock cannot pin down. Every function it calls is here.

//! The n-gram table reader: 320M rows that never live in RAM, every row off an
//! unbuffered 4 KiB read.
//!
//! `issue` builds the jobs (dedup by aligned page, the 2-page window for a row
//! that straddles a page boundary, the sort by offset so a prefill chunk reads
//! near-sequentially), `collect` waits for them. The row cache is not the table:
//! it is a bounded set-associative cache of rows this process already fetched,
//! and capacity 0 disables it.
//!
//! The platform layer is a trait (`Io`) rather than a link dependency, which is
//! what lets the whole thing run against a scripted file on a machine with no
//! SSD. `strata-device` implements it over the ABI's `file_*` slots; the corpus
//! test implements it over an in-memory image with a scripted clock.
//!
//! The properties the C++ got right and this keeps:
//!
//! * Out-of-range rows produce zero bytes, not an error - that is the mmap path's
//!   behaviour and callers depend on it. They are also never deduped: they do not
//!   reach the page map.
//! * A cache hit is served inside `issue` and never becomes a job, so it does not
//!   count as a read and does not hold the ticket open.
//! * `pending` counts jobs, not rows. A ticket whose rows all came from the cache
//!   or from out-of-range has `pending == 0` and `collect` returns at once.
//! * A short read is a refusal, not a zero fill: `in_page + row_bytes > bytes`
//!   means the row the caller asked for is not all there.
//! * `cancel_queued` decrements the pending count of everything still queued, so a
//!   failed read releases the tickets waiting on it instead of hanging them.
//! * The completion tag is range-checked in `process`, and nowhere else -
//!   `release_delayed` indexes `inflight` unguarded because nothing reaches the
//!   delayed list without having passed that check.
//! * `close()` drains outstanding reads by handing their slots back, without
//!   applying them: no output write, no cache insert, no stats. Whatever was in
//!   flight when the reader closed is simply gone.
//! * `close()` does not reset the error, the injected delay, the ticket counter,
//!   the ring position, the stats or the rng. It does reset the cache. A reader
//!   reopened on the same instance keeps its counters, and the golden pins that.
//!
//! One deliberate difference from the C++: `issue` takes an `OutId` registered on
//! the reader instead of a caller-owned buffer. The C++ documents "out_raw must
//! stay valid until collect returns" in prose; here the reader owns the buffer, so
//! the borrow checker owns that contract.

use std::collections::{BTreeMap, HashMap, VecDeque};

pub const ROW_BYTES: u32 = 90;
pub const PAGE: u64 = 4096;
/// A `wake()` shows up as one completion with this tag.
pub const WAKE_TAG: u64 = u64::MAX;

const LATENCY_RING: usize = 65_536;
const WAYS: usize = 8;
const EMPTY: u32 = 0xFFFF_FFFF;

/// What the platform layer owes this reader. `strata-device` implements it over the
/// ABI; the read buffers live on the implementation side so the reader never holds a
/// pointer into someone else's memory.
pub trait Io {
    fn open(&mut self, path: &str) -> Result<(), String>;
    fn close(&mut self);
    fn is_open(&self) -> bool;
    fn size(&self) -> u64;
    fn alignment(&self) -> u64 {
        PAGE
    }
    /// The reader asks for `max_inflight * 2 * PAGE` at `open` and hands the
    /// implementation back its own slot index.
    fn alloc_slab(&mut self, bytes: usize) -> bool;
    fn free_slab(&mut self);
    /// Copy `out.len()` bytes out of the buffer `alloc_slab` handed out for `slot`,
    /// starting at `from` within it.
    fn slot_read(&mut self, slot: usize, from: usize, out: &mut [u8]);
    /// Queue one read into `slot`. A queued read produces exactly one completion.
    fn submit(&mut self, offset: u64, slot: usize, length: u32, tag: u64) -> Result<(), String>;
    /// The completions ready by the scripted clock, up to `max`. `timeout_ms < 0`
    /// blocks until at least one is; `0` polls.
    fn wait(&mut self, out: &mut Vec<Completion>, max: usize, timeout_ms: i32);
    fn wake(&mut self);
    fn now_us(&self) -> f64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completion {
    pub tag: u64,
    pub bytes: u32,
    pub ok: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReaderStats {
    /// rows asked for
    pub requests: u64,
    /// rows served from the row cache
    pub cache_hits: u64,
    /// rows that shared a page already being read in the same ticket
    pub dedup_rows: u64,
    /// SSD read requests issued
    pub reads: u64,
    /// bytes read from the SSD
    pub bytes: u64,
    /// time `collect` spent blocked
    pub wait_us: f64,
    /// time spent inside the read submission call (non-zero = it blocks)
    pub submit_us: f64,
    /// sum of per-read latencies (issue to completion)
    pub read_us_sum: f64,
    /// reads delayed by fault injection
    pub late_injected: u64,
    /// pages read only to keep the SSD awake (not in `reads`, `bytes` or latencies)
    pub keepalive_reads: u64,
    /// the slowest of them
    pub keepalive_us_max: f64,
    /// last <= 65,536 read latencies, for percentiles
    pub read_us: Vec<f32>,
}

impl ReaderStats {
    /// The `q`-th percentile. `nth_element` in the C++; the k-th order statistic,
    /// which is the same number.
    #[must_use]
    pub fn percentile(&self, q: f64) -> f64 {
        if self.read_us.is_empty() {
            return 0.0;
        }
        let n = self.read_us.len();
        let k = std::cmp::min(n - 1, (q * (n - 1) as f64 + 0.5) as usize);
        let mut v = self.read_us.clone();
        v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        f64::from(v[k])
    }
}

/// Set-associative row cache: 8 ways per set, round-robin replacement inside a set.
/// Bounded by construction.
#[derive(Debug, Default)]
pub struct RowCache {
    pub sets: u64,
    pub rb: u32,
    keys: Vec<u32>,
    data: Vec<u8>,
    next: Vec<u8>,
    used: u64,
}

impl RowCache {
    fn init(&mut self, rows: u64, row_bytes: u32) {
        self.sets = rows / WAYS as u64;
        self.rb = row_bytes;
        self.keys = vec![EMPTY; self.sets as usize * WAYS];
        self.data = vec![0; self.sets as usize * WAYS * row_bytes as usize];
        self.next = vec![0; self.sets as usize];
        self.used = 0;
    }

    #[must_use]
    pub fn mix(r: u32) -> u64 {
        let x = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^ (x >> 29)
    }

    fn find(&self, row: u32) -> Option<usize> {
        if self.sets == 0 {
            return None;
        }
        let s = Self::mix(row) % self.sets;
        for w in 0..WAYS {
            if self.keys[s as usize * WAYS + w] == row {
                return Some((s as usize * WAYS + w) * self.rb as usize);
            }
        }
        None
    }

    fn insert(&mut self, row: u32, bytes: &[u8]) {
        if self.sets == 0 || self.find(row).is_some() {
            return;
        }
        let s = (Self::mix(row) % self.sets) as usize;
        let w = self.next[s] as usize;
        self.next[s] = ((w + 1) % WAYS) as u8;
        if self.keys[s * WAYS + w] == EMPTY {
            self.used += 1;
        }
        self.keys[s * WAYS + w] = row;
        let off = (s * WAYS + w) * self.rb as usize;
        self.data[off..off + self.rb as usize].copy_from_slice(bytes);
    }
}

#[derive(Debug, Clone, Copy)]
struct Use {
    row: u32,
    in_page: u32,
    dst: usize, // byte offset into the ticket's output buffer
}

#[derive(Debug, Clone, Default)]
struct Job {
    offset: u64, // aligned file offset
    length: u32, // PAGE, or 2 * PAGE for a row that straddles a page boundary
    ticket: u32,
    uses: Vec<Use>,
    issued_us: f64,
    keepalive: bool, // no rows and no ticket: it only keeps the SSD awake
}

#[derive(Debug, Default)]
struct TicketState {
    pending: u32, // jobs not yet completed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct OutId(pub u32);

pub struct PleReader<I: Io> {
    io: I,
    table_offset: u64,
    n_rows: u64,
    row_bytes: u32,
    max_inflight: u32,
    free_slots: Vec<u32>,
    inflight: Vec<Job>,
    delayed: Vec<Completion>,
    queue: VecDeque<Job>,
    tickets: HashMap<u32, TicketState>,
    next_ticket: u32,
    cache: RowCache,
    stats: ReaderStats,
    ring_pos: usize,
    delay_us: f64,
    error: String,
    keep_us: f64,
    keep_window_us: f64,
    last_issue_us: f64,
    last_read_us: f64,
    rng: u64,
    /// The C++ worker loop is not ported; this reader is the caller-thread arm. The
    /// flag stays because it gates the keep-alive.
    threaded: bool,
    outputs: BTreeMap<OutId, Vec<u8>>,
    next_out: u32,
}

impl<I: Io> PleReader<I> {
    #[must_use]
    pub fn new(io: I) -> PleReader<I> {
        PleReader {
            io,
            table_offset: 0,
            n_rows: 0,
            row_bytes: ROW_BYTES,
            max_inflight: 0,
            free_slots: Vec::new(),
            inflight: Vec::new(),
            delayed: Vec::new(),
            queue: VecDeque::new(),
            tickets: HashMap::new(),
            next_ticket: 1,
            cache: RowCache::default(),
            stats: ReaderStats::default(),
            ring_pos: 0,
            delay_us: 0.0,
            error: String::new(),
            keep_us: 0.0,
            keep_window_us: 0.0,
            last_issue_us: 0.0,
            last_read_us: 0.0,
            rng: 0,
            threaded: false,
            outputs: BTreeMap::new(),
            next_out: 1,
        }
    }

    pub fn open(
        &mut self,
        path: &str,
        table_offset: u64,
        n_rows: u64,
        max_inflight: u32,
        cache_rows: u64,
        row_bytes: u32,
    ) -> Result<(), String> {
        self.close();
        if max_inflight == 0 || max_inflight > 1024 {
            return Err("PleReader: max_inflight must be 1..1024".to_string());
        }
        if row_bytes == 0 || row_bytes > PAGE as u32 {
            return Err("PleReader: row_bytes must be 1..4096".to_string());
        }
        self.io.open(path)?;
        self.row_bytes = row_bytes;
        if table_offset + n_rows * u64::from(row_bytes) > self.io.size() {
            self.close();
            return Err(format!(
                "PleReader: the table extends past the end of {path}"
            ));
        }
        self.table_offset = table_offset;
        self.n_rows = n_rows;
        self.max_inflight = max_inflight;
        if !self
            .io
            .alloc_slab(max_inflight as usize * 2 * PAGE as usize)
        {
            self.close();
            return Err("PleReader: cannot allocate read buffers".to_string());
        }
        self.inflight = vec![Job::default(); max_inflight as usize];
        self.free_slots = (0..max_inflight).rev().collect();
        self.cache.init(cache_rows, row_bytes);
        self.error.clear();
        self.keep_us = 0.0;
        self.last_issue_us = 0.0;
        self.last_read_us = 0.0;
        self.rng = 0x9E37_79B9_7F4A_7C15 ^ (self.io.now_us() as u64);
        if self.rng == 0 {
            self.rng = 1;
        }
        self.reset_stats();
        self.threaded = false;
        Ok(())
    }

    pub fn close(&mut self) {
        // Outstanding reads must finish before their buffers are released.
        if self.io.is_open() {
            while (self.free_slots.len() as u32) < self.max_inflight {
                let mut got = Vec::new();
                self.io.wait(&mut got, 64, -1);
                if got.is_empty() {
                    break;
                }
                for c in got {
                    if c.tag != WAKE_TAG {
                        self.free_slots.push(c.tag as u32);
                    }
                }
            }
        }
        self.io.close();
        self.io.free_slab();
        self.queue.clear();
        self.tickets.clear();
        self.delayed.clear();
        self.inflight.clear();
        self.free_slots.clear();
        self.cache.init(0, self.row_bytes);
        self.threaded = false;
        self.keep_us = 0.0;
        self.last_issue_us = 0.0;
        self.last_read_us = 0.0;
    }

    #[must_use]
    pub fn is_open(&self) -> bool {
        self.io.is_open()
    }

    #[must_use]
    pub fn row_bytes(&self) -> u32 {
        self.row_bytes
    }

    /// Register an output buffer for `rows` rows. The reader owns it, which is what
    /// makes the C++'s "must stay valid until collect returns" a compile-time fact.
    pub fn register_output(&mut self, rows: usize) -> OutId {
        let id = OutId(self.next_out);
        self.next_out += 1;
        self.outputs
            .insert(id, vec![0; rows * self.row_bytes as usize]);
        id
    }

    #[must_use]
    pub fn output(&self, id: OutId) -> Option<&[u8]> {
        self.outputs.get(&id).map(Vec::as_slice)
    }

    #[must_use]
    pub fn output_mut(&mut self, id: OutId) -> Option<&mut Vec<u8>> {
        self.outputs.get_mut(&id)
    }

    pub fn set_injected_delay_us(&mut self, delay_us: f64) {
        self.delay_us = if delay_us < 0.0 { 0.0 } else { delay_us };
    }

    pub fn set_keepalive(&mut self, period_ms: f64, window_s: f64) {
        self.keep_us = if self.threaded && period_ms > 0.0 {
            period_ms * 1000.0
        } else {
            0.0
        };
        self.keep_window_us = if window_s > 0.0 { window_s * 1e6 } else { 0.0 };
    }

    #[must_use]
    pub fn stats(&self) -> &ReaderStats {
        &self.stats
    }

    #[must_use]
    pub fn snapshot(&self) -> ReaderStats {
        self.stats.clone()
    }

    pub fn reset_stats(&mut self) {
        self.stats = ReaderStats::default();
        self.ring_pos = 0;
    }

    #[must_use]
    pub fn cache_capacity(&self) -> u64 {
        self.cache.sets * WAYS as u64
    }

    #[must_use]
    pub fn cache_size(&self) -> u64 {
        self.cache.used
    }

    /// The cache on its own, for the corpus: what a lookup would return, and which set
    /// the row would land in.
    #[must_use]
    pub fn cache_probe(&self, row: u32) -> (bool, u64) {
        let hit = self.cache.find(row).is_some();
        let set = if self.cache.sets == 0 {
            0
        } else {
            RowCache::mix(row) % self.cache.sets
        };
        (hit, set)
    }

    /// Insert `row` into the cache from `bytes`, exactly as a completion would.
    pub fn cache_insert(&mut self, row: u32, bytes: &[u8]) {
        self.cache.insert(row, bytes);
    }

    /// The slots `free_slots` would hand out, in order: a stack seeded high to low, so
    /// the first one used is the last one.
    #[must_use]
    pub fn free_slots(&self) -> &[u32] {
        &self.free_slots
    }

    #[must_use]
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    #[must_use]
    pub fn inflight_len(&self) -> usize {
        self.inflight.len()
    }

    // ---- the corpus surface: the pieces the C++ harness reaches through
    // `#define private public`, exposed here so the replay can drive them too.

    #[must_use]
    pub fn io(&self) -> &I {
        &self.io
    }

    pub fn io_mut(&mut self) -> &mut I {
        &mut self.io
    }

    /// The C++ worker loop is not ported; this sets the flag that gates the
    /// keep-alive without starting one.
    pub fn set_threaded(&mut self, b: bool) {
        self.threaded = b;
    }

    #[must_use]
    pub fn error(&self) -> &str {
        &self.error
    }

    pub fn set_error(&mut self, s: &str) {
        self.error = s.to_string();
    }

    #[must_use]
    pub fn keep_window_us(&self) -> f64 {
        self.keep_window_us
    }

    #[must_use]
    pub fn queue_back(&self) -> (u64, u32) {
        let j = self.queue.back().expect("queue empty");
        (j.offset, j.length)
    }

    #[must_use]
    pub fn ring_pos(&self) -> usize {
        self.ring_pos
    }

    pub fn set_ring_pos(&mut self, n: usize) {
        self.ring_pos = n;
    }

    pub fn clear_read_us(&mut self) {
        self.stats.read_us.clear();
    }

    pub fn record_latency(&mut self, us: f64) {
        self.record_latency_us(us);
    }

    #[must_use]
    pub fn cache_sets(&self) -> u64 {
        self.cache.sets
    }

    #[must_use]
    pub fn cache_rb(&self) -> u32 {
        self.cache.rb
    }

    /// Arm the keep-alive clock the way an `issue` would have.
    pub fn arm(&mut self, now: f64) {
        self.last_issue_us = now;
        self.last_read_us = now;
    }

    fn dec_pending(&mut self, ticket: u32) {
        if let Some(ts) = self.tickets.get_mut(&ticket) {
            if ts.pending > 0 {
                ts.pending -= 1;
            }
        }
    }

    fn cancel_queued(&mut self) {
        let tickets: Vec<u32> = self.queue.iter().map(|j| j.ticket).collect();
        self.queue.clear();
        for t in tickets {
            self.dec_pending(t);
        }
    }

    fn record_latency_us(&mut self, us: f64) {
        self.stats.read_us_sum += us;
        if self.stats.read_us.len() < LATENCY_RING {
            self.stats.read_us.push(us as f32);
        } else {
            self.stats.read_us[self.ring_pos % LATENCY_RING] = us as f32;
            self.ring_pos += 1;
        }
    }

    fn pump(&mut self) -> bool {
        while !self.queue.is_empty() && !self.free_slots.is_empty() {
            let s = *self.free_slots.last().unwrap() as usize;
            self.free_slots.pop();
            self.inflight[s] = self.queue.pop_front().unwrap();
            self.inflight[s].issued_us = self.io.now_us();
            let keepalive = self.inflight[s].keepalive;
            let (offset, length) = (self.inflight[s].offset, self.inflight[s].length);
            if let Err(e) = self.io.submit(offset, s, length, s as u64) {
                // a keep-alive read that cannot go out must not fail the reader
                if keepalive {
                    self.keep_us = 0.0;
                    self.inflight[s].keepalive = false;
                    self.free_slots.push(s as u32);
                    continue;
                }
                self.error = e;
                self.inflight[s].uses.clear();
                let t = self.inflight[s].ticket;
                self.dec_pending(t);
                self.free_slots.push(s as u32);
                self.cancel_queued();
                return false;
            }
            let issued = self.inflight[s].issued_us;
            self.last_read_us = issued;
            if !keepalive {
                self.stats.submit_us += self.io.now_us() - issued;
                self.stats.reads += 1;
            }
        }
        true
    }

    /// When the next keep-alive read is due, or < 0 when none is: it is off, the reader
    /// failed, or no rows were asked for within the window (then the SSD may sleep; the
    /// next `issue` re-arms it).
    #[must_use]
    pub fn keepalive_due(&self, now: f64) -> f64 {
        if self.keep_us <= 0.0
            || !self.error.is_empty()
            || self.last_issue_us <= 0.0
            || now - self.last_issue_us > self.keep_window_us
        {
            return -1.0;
        }
        self.last_read_us + self.keep_us
    }

    /// One page of the table, a different one each time, so the SSD really reads (not
    /// its controller's buffer).
    pub fn queue_keepalive(&mut self) {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let first = self.table_offset / PAGE;
        let end = (self.table_offset + self.n_rows * u64::from(self.row_bytes)) / PAGE;
        self.queue.push_back(Job {
            offset: (first
                + if end > first {
                    self.rng % (end - first)
                } else {
                    0
                })
                * PAGE,
            length: PAGE as u32,
            keepalive: true,
            ..Job::default()
        });
    }

    fn finish(&mut self, c: &Completion) -> bool {
        let s = c.tag as usize;
        if s >= self.inflight.len() {
            self.error = "PleReader: invalid table-read completion".to_string();
            self.cancel_queued();
            return false;
        }
        let rb = self.row_bytes as usize;
        if self.inflight[s].keepalive {
            // no rows: it counts once it is done, with how long it took; a failed one
            // turns the keep-alive off, not the reader
            if c.ok {
                self.stats.keepalive_reads += 1;
                let lat = self.io.now_us() - self.inflight[s].issued_us;
                self.stats.keepalive_us_max = self.stats.keepalive_us_max.max(lat);
            } else {
                self.keep_us = 0.0;
            }
            self.inflight[s].keepalive = false;
            self.free_slots.push(s as u32);
            return if self.error.is_empty() {
                self.pump()
            } else {
                true
            };
        }
        if !c.ok {
            self.error = "PleReader: a table read failed".to_string();
            self.cancel_queued();
            self.inflight[s].uses.clear();
            let t = self.inflight[s].ticket;
            self.dec_pending(t);
            self.free_slots.push(s as u32);
            return false;
        }
        let lat = self.io.now_us() - self.inflight[s].issued_us;
        self.record_latency_us(lat);
        self.stats.bytes += u64::from(c.bytes);
        for u in &self.inflight[s].uses {
            if u.in_page as usize + rb > c.bytes as usize {
                self.error = "PleReader: short read inside the table".to_string();
                self.cancel_queued();
                self.inflight[s].uses.clear();
                let t = self.inflight[s].ticket;
                self.dec_pending(t);
                self.free_slots.push(s as u32);
                return false;
            }
        }
        // The slot belongs to the Io and the output and the cache belong to the reader,
        // so each row comes across through a copy - the C++ memcpy is the same copy.
        let uses = self.inflight[s].uses.clone();
        let mut gathered: Vec<(usize, u32, Vec<u8>)> = Vec::with_capacity(uses.len());
        for u in &uses {
            let mut bytes = vec![0u8; rb];
            self.io.slot_read(s, u.in_page as usize, &mut bytes);
            gathered.push((u.dst, u.row, bytes));
        }
        let ticket = self.inflight[s].ticket;
        for (dst, row, bytes) in gathered {
            if let Some(out) = self.outputs.get_mut(&OutId(ticket)) {
                out[dst..dst + rb].copy_from_slice(&bytes);
            }
            self.cache.insert(row, &bytes);
        }
        self.dec_pending(ticket);
        self.inflight[s].uses.clear();
        self.free_slots.push(s as u32);
        if self.error.is_empty() {
            self.pump()
        } else {
            true
        }
    }

    /// Completions (or wake packets) just returned by the Io, applying fault injection.
    pub fn process(&mut self, got: &[Completion]) -> bool {
        let mut ok = true;
        for c in got {
            if c.tag == WAKE_TAG {
                continue;
            }
            if c.tag as usize >= self.inflight.len() {
                self.error = "PleReader: invalid table-read completion".to_string();
                self.cancel_queued();
                ok = false;
                continue;
            }
            let lat = self.io.now_us() - self.inflight[c.tag as usize].issued_us;
            if self.delay_us > 0.0 && lat < self.delay_us {
                self.delayed.push(*c);
                self.stats.late_injected += 1;
                continue;
            }
            if !self.finish(c) {
                ok = false; // drain the rest of this batch so no completed slot is stranded
            }
        }
        ok
    }

    pub fn release_delayed(&mut self) -> bool {
        let now = self.io.now_us();
        let mut i = 0;
        while i < self.delayed.len() {
            let lat = now - self.inflight[self.delayed[i].tag as usize].issued_us;
            if lat >= self.delay_us {
                let c = self.delayed.remove(i);
                if !self.finish(&c) {
                    return false;
                }
            } else {
                i += 1;
            }
        }
        true
    }

    /// Process whatever has completed; blocks up to `timeout_ms` for the first
    /// completion.
    pub fn drain(&mut self, timeout_ms: i32) -> bool {
        let mut timeout = timeout_ms;
        if !self.delayed.is_empty() {
            if !self.release_delayed() {
                return false;
            }
            timeout = 0; // keep polling the held completions
        }
        let mut got = Vec::new();
        self.io.wait(&mut got, 64, timeout);
        self.process(&got)
    }

    /// Start fetching `n` rows into the registered output buffer. Out-of-range rows
    /// produce zero bytes.
    pub fn issue(&mut self, rows: &[u32], out: OutId) -> Ticket {
        let id = self.next_ticket;
        self.next_ticket += 1;
        if self.next_ticket == 0 {
            self.next_ticket = 1;
        }
        let now = self.io.now_us();
        let rearm = self.keep_us > 0.0
            && (self.last_issue_us <= 0.0 || now - self.last_issue_us > self.keep_window_us);
        self.last_issue_us = now;
        let rb = self.row_bytes as usize;
        let mut by_page: HashMap<u64, usize> = HashMap::new();
        let mut jobs: Vec<Job> = Vec::new();
        for (i, &row) in rows.iter().enumerate() {
            let dst = i * rb;
            self.stats.requests += 1;
            if row >= self.n_rows as u32 {
                self.out_fill(out, dst, None);
                continue;
            }
            let hit = self
                .cache
                .find(row)
                .map(|off| self.cache.data[off..off + rb].to_vec());
            if let Some(bytes) = hit {
                self.out_fill(out, dst, Some(&bytes));
                self.stats.cache_hits += 1;
                continue;
            }
            let at = self.table_offset + u64::from(row) * rb as u64;
            let first = at / PAGE * PAGE;
            let length = ((at + rb as u64 - 1) / PAGE * PAGE - first + PAGE) as u32;
            if let Some(&idx) = by_page.get(&first) {
                let j = &mut jobs[idx];
                j.length = std::cmp::max(j.length, length);
                j.uses.push(Use {
                    row,
                    in_page: (at - first) as u32,
                    dst,
                });
                self.stats.dedup_rows += 1;
                continue;
            }
            by_page.insert(first, jobs.len());
            jobs.push(Job {
                offset: first,
                length,
                ticket: id,
                uses: vec![Use {
                    row,
                    in_page: (at - first) as u32,
                    dst,
                }],
                ..Job::default()
            });
        }
        // Sorted by offset: prefill chunks then read the SSD in near-sequential order.
        jobs.sort_unstable_by_key(|j| j.offset);
        let pending = jobs.len() as u32;
        self.tickets.entry(id).or_default().pending = pending;
        for j in jobs {
            self.queue.push_back(j);
        }
        if !self.pump() && self.error.is_empty() {
            self.error = "PleReader: submit failed".to_string();
        }
        let _ = rearm; // the C++ wakes the worker here; there is no worker on this arm
        Ticket(id)
    }

    /// Block until every row of the ticket is in its output buffer.
    pub fn collect(&mut self, t: Ticket) -> Result<(), String> {
        let start = self.io.now_us();
        let mut pending = self.ticket_pending(t.0)?;
        while pending > 0 {
            if !self.drain(-1) && self.error.is_empty() {
                self.error = "PleReader: read failed".to_string();
            }
            pending = self.ticket_pending(t.0)?;
        }
        if !self.error.is_empty() {
            return Err(self.error.clone());
        }
        self.stats.wait_us += self.io.now_us() - start;
        self.tickets.remove(&t.0);
        Ok(())
    }

    fn ticket_pending(&self, id: u32) -> Result<u32, String> {
        self.tickets
            .get(&id)
            .map(|ts| ts.pending)
            .ok_or_else(|| "PleReader: unknown ticket".to_string())
    }

    fn out_fill(&mut self, out: OutId, dst: usize, bytes: Option<&[u8]>) {
        let rb = self.row_bytes as usize;
        let Some(buf) = self.outputs.get_mut(&out) else {
            return;
        };
        match bytes {
            Some(b) => buf[dst..dst + rb].copy_from_slice(b),
            None => buf[dst..dst + rb].fill(0),
        }
    }
}
