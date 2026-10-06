// @generated: ported from Strata src/core/expert_cache.cpp - the whole file. The
// device layer is the seam; everything above it is deterministic and the corpus
// covers it.

//! The VRAM-resident expert tier: the slot storage and the residency table.
//!
//! It allocates the arena, fills it from the host, and answers
//! `(layer, expert) -> slot or -1`. It computes nothing itself.
//!
//! The device layer is a trait (`Device`) rather than a link dependency, which is
//! what lets the whole cache run against a scripted device on a machine with no
//! GPU. `strata-device` implements it over the ABI; the corpus test implements it
//! over host memory. The core holds device memory as an opaque address and never
//! dereferences it - the seam does the copying.
//!
//! The properties the C++ got right and this keeps:
//!
//! * The allocation is checked against the card BEFORE it is made, and a card that
//!   will not answer (`cudaMemGetInfo` fails) skips the check and allocates anyway.
//!   A cache that silently took less than it was asked for would report a hit rate
//!   for slots it does not have.
//! * `admit` never evicts. Eviction policy is a measured question and a placeholder
//!   policy would set the hit rate everything downstream is sized against.
//! * Per-layer admission is off by default, and the global path does not touch the
//!   `admitted` counter - `resident()` reads `next_free` there and `admitted` under
//!   per-layer admission, which is why the two paths report differently for the same
//!   number of claims.
//! * The last layer takes the remainder of the division, so the per-layer ranges
//!   cover the arena exactly and no slot is orphaned.
//! * `shrink` rounds UP to a segment and `grow` rounds DOWN; a shrink that would
//!   keep everything it already has returns without touching the device.
//! * A segment that was unmapped comes back as NEW physical memory, not the old
//!   bytes - the driver hands out a fresh allocation, so the caller has to refill it.
//!   Nothing reads a slot while its segment is unmapped: the address range has no
//!   backing then, and what a read would return is the driver's business.
//! * `close()` frees the arena and clears the table. A reader reopened on the same
//!   instance starts empty.
//! * `verify_slot` is the only thing that says the cache holds the expert it claims
//!   to, so it reads back through the seam rather than trusting the table.
//!
//! The HIP-only blocking-staging buffer (`#ifdef STRATA_USE_HIP`) is compiled out of
//! the build the golden was made from and is NOT covered by it; in Rust it belongs to
//! the device layer, which may stage a pageable source however it likes. The gfx906
//! build, where the segmented cache refuses at open, is likewise not covered.

/// `(layer, expert)` -> slot, or this.
pub const NOT_RESIDENT: i32 = -1;

/// What the device layer owes this cache. Every method is a CUDA runtime or driver
/// call; the error strings are what those calls report, because the messages the C++
/// printed embed them.
pub trait Device {
    /// `cudaMemGetInfo`: `None` when the card will not answer, which is not an error -
    /// the caller skips the check and allocates anyway.
    fn mem_get_info(&mut self) -> Option<(u64, u64)>;
    fn alloc(&mut self, bytes: u64) -> Result<u64, String>;
    fn free(&mut self, addr: u64);
    fn memset(&mut self, addr: u64, value: u8, bytes: u64) -> Result<(), String>;
    /// The stream-ordered copy: `fill_slot` on the caller's stream,
    /// `fill_slot_queued` on the legacy one (stream 0).
    fn memcpy_h2d_async(&mut self, dst: u64, src: &[u8], stream: u64) -> Result<(), String>;
    /// The blocking copy: `fill_slot_blocking`. Kept separate because that is what
    /// the startup path needs - a legacy-stream copy is not ordered against a
    /// non-blocking one, and a check that can be read before it has happened is not
    /// a check.
    fn memcpy_h2d_sync(&mut self, dst: u64, src: &[u8]) -> Result<(), String>;
    fn memcpy_d2h(&mut self, dst: &mut [u8], src: u64) -> Result<(), String>;
    fn sync_device(&mut self) -> Result<(), String>;
    fn sync_stream(&mut self, stream: u64) -> Result<(), String>;

    /// The driver's virtual memory management: entry-point lookup, the device
    /// attribute and the granularity all have to succeed for a segmented arena, so
    /// the seam reports them as one question.
    fn vmm_supported(&mut self) -> bool;
    fn granularity(&mut self) -> Option<u64>;
    fn address_reserve(&mut self, bytes: u64) -> Option<u64>;
    fn address_free(&mut self, addr: u64, bytes: u64);
    fn mem_create(&mut self, bytes: u64) -> Option<u64>;
    fn mem_release(&mut self, handle: u64) -> bool;
    fn mem_map(&mut self, addr: u64, bytes: u64, handle: u64) -> bool;
    fn mem_unmap(&mut self, addr: u64, bytes: u64) -> bool;
    fn mem_set_access(&mut self, addr: u64, bytes: u64) -> bool;
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1073741824.0
}

// ---- the profile file format (tools/make_profile.py's, little-endian)

/// Read `profile.bin`: `STRP`, `version, n_layers, n_expert, slots, n_ranked`, then
/// the ranked pairs. The frequency table is the profile's own working and is not
/// returned - the engine measures the hit rate by running, not by re-deriving what
/// the file claims.
pub fn read_expert_profile(
    path: &str,
    n_layers: i64,
    n_expert: i64,
) -> Result<(Vec<(i32, i32)>, i64), String> {
    let bytes =
        std::fs::read(path).map_err(|_| format!("read_expert_profile: cannot open {path}"))?;
    if bytes.len() < 24 {
        return Err(format!(
            "read_expert_profile: {path} is too short to hold a header"
        ));
    }
    if &bytes[0..4] != b"STRP" {
        return Err(format!(
            "read_expert_profile: {path} does not start with STRP"
        ));
    }
    let hdr =
        |i: usize| -> u32 { u32::from_le_bytes(bytes[4 + i * 4..8 + i * 4].try_into().unwrap()) };
    let (nl, ne) = (hdr(1) as i64, hdr(2) as i64);
    let (want, n_ranked) = (hdr(3), hdr(4));
    if nl != n_layers || ne != n_expert {
        return Err(format!(
            "read_expert_profile: {path} is {nl}x{ne} but this model is {n_layers}x{n_expert} - it is a \
             profile for a different artifact"
        ));
    }
    if i64::from(n_ranked) > i64::from(want) {
        return Err("read_expert_profile: the header claims more ranked pairs than slots".into());
    }
    let body = &bytes[24..];
    if body.len() < (n_ranked as usize) * 4 {
        return Err("read_expert_profile: the ranked list is truncated".into());
    }
    let mut ranked = Vec::with_capacity(n_ranked as usize);
    for i in 0..n_ranked {
        let l =
            u16::from_le_bytes(body[i as usize * 4..i as usize * 4 + 2].try_into().unwrap()) as i32;
        let e = u16::from_le_bytes(
            body[i as usize * 4 + 2..i as usize * 4 + 4]
                .try_into()
                .unwrap(),
        ) as i32;
        if l < 0 || i64::from(l) >= n_layers || e < 0 || i64::from(e) >= n_expert {
            return Err(format!(
                "read_expert_profile: pair {i} is (layer {l}, expert {e}), out of range"
            ));
        }
        ranked.push((l, e));
    }
    // `version` is read and unused: a future format bumps it, and the layout check
    // above is what protects this reader today.
    Ok((ranked, i64::from(want)))
}

/// What the adaptive tier learned, as a profile ranking EVERY `(layer, expert)` pair:
/// the ones resident now first (where the swaps left the cache), then the rest;
/// within each, by heat descending, then by prior rank (an expert this run never
/// routed keeps its old place), then by index. A start from it begins where this one
/// ended; one with more slots adds the hottest of the rest, one with fewer keeps the
/// hottest.
///
/// `resident` and `heat` are `n_layers * n_expert` entries; past the end of either,
/// an entry reads as not-resident and heat 0.
pub fn rank_learned_profile(
    n_layers: i64,
    n_expert: i64,
    resident: &[u8],
    heat: &[f64],
    prior: &[(i32, i32)],
) -> Vec<(i32, i32)> {
    let n = (n_layers * n_expert) as usize;
    const UNRANKED: i64 = i64::MAX;
    let mut prior_rank = vec![UNRANKED; n];
    for (r, &(l, e)) in prior.iter().enumerate() {
        if l >= 0 && i64::from(l) < n_layers && e >= 0 && i64::from(e) < n_expert {
            let slot = &mut prior_rank[(i64::from(l) * n_expert + i64::from(e)) as usize];
            if *slot == UNRANKED {
                *slot = r as i64;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    let res = |i: usize| resident.get(i).copied().unwrap_or(0) != 0;
    let ht = |i: usize| heat.get(i).copied().unwrap_or(0.0);
    // The C++ compares doubles with `!=`, so a NaN is "different" and the order it
    // produces is the sort's business, not the comparator's. The corpus has no NaNs.
    order.sort_by(|&a, &b| {
        if res(a) != res(b) {
            return if res(a) {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
        if ht(a) != ht(b) {
            return if ht(a) > ht(b) {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
        if prior_rank[a] != prior_rank[b] {
            return prior_rank[a].cmp(&prior_rank[b]);
        }
        a.cmp(&b)
    });
    order
        .into_iter()
        .map(|i| ((i as i64 / n_expert) as i32, (i as i64 % n_expert) as i32))
        .collect()
}

/// Write `ranked` in `tools/make_profile.py`'s format, byte for byte: `STRP`,
/// `1, n_layers, n_expert, n_ranked, n_ranked` (uint32), the pairs (uint16 layer,
/// uint16 expert), then the `n_layers x n_expert` int32 table of each pair's rank
/// (-1: not ranked). Atomically: a `<path>.tmp` beside it, renamed over `path` once
/// complete, so a reader (the next start) never sees half a file.
pub fn write_expert_profile(
    path: &str,
    n_layers: i64,
    n_expert: i64,
    ranked: &[(i32, i32)],
) -> Result<(), String> {
    if n_layers <= 0 || n_expert <= 0 || n_layers > 65535 || n_expert > 65535 {
        return Err("write_expert_profile: the model's layout does not fit the format".into());
    }
    let mut table = vec![-1i32; (n_layers * n_expert) as usize];
    let mut pairs: Vec<u8> = Vec::with_capacity(ranked.len() * 4);
    for (r, &(l, e)) in ranked.iter().enumerate() {
        if l < 0 || i64::from(l) >= n_layers || e < 0 || i64::from(e) >= n_expert {
            return Err("write_expert_profile: a ranked pair is out of range".into());
        }
        table[(i64::from(l) * n_expert + i64::from(e)) as usize] = r as i32;
        pairs.extend_from_slice(&(l as u16).to_le_bytes());
        pairs.extend_from_slice(&(e as u16).to_le_bytes());
    }
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"STRP");
    for v in [
        1u32,
        n_layers as u32,
        n_expert as u32,
        ranked.len() as u32,
        ranked.len() as u32,
    ] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&pairs);
    for t in table {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    // the format is little-endian (make_profile.py's "<"): so is every machine this engine runs on
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, &bytes)
        .map_err(|_| format!("write_expert_profile: cannot create {tmp}"))?;
    match std::fs::rename(&tmp, path) {
        // replaces an existing file (MoveFileEx / rename(2))
        Ok(()) => Ok(()),
        Err(_) => {
            let _ = std::fs::remove_file(&tmp);
            Err(format!("write_expert_profile: cannot write {path}"))
        }
    }
}

// ---- the cache

pub struct ExpertCache<D: Device> {
    dev: D,
    base: u64,
    live_slots: i64,
    seg_req: i64,
    seg: i64,
    reserved: u64,
    segs: Vec<u64>,
    seg_size: Vec<i64>,
    mapped_segs: i64,
    residency: Vec<i32>,
    allocated: i64,
    n_layers: i64,
    n_expert: i64,
    blob: i64,
    next_free: i64,
    fills: i64,
    per_layer: bool,
    layer_next: Vec<i32>,
    off: Vec<u64>,
    admitted: i64,
}

impl<D: Device> ExpertCache<D> {
    pub fn new(dev: D) -> Self {
        ExpertCache {
            dev,
            base: 0,
            live_slots: 0,
            seg_req: 0,
            seg: 0,
            reserved: 0,
            segs: Vec::new(),
            seg_size: Vec::new(),
            mapped_segs: 0,
            residency: Vec::new(),
            allocated: 0,
            n_layers: 0,
            n_expert: 0,
            blob: 0,
            next_free: 0,
            fills: 0,
            per_layer: false,
            layer_next: Vec::new(),
            off: Vec::new(),
            admitted: 0,
        }
    }

    /// The device layer is the caller's; hand it back for its own bookkeeping.
    pub fn device_mut(&mut self) -> &mut D {
        &mut self.dev
    }

    pub fn set_segment_bytes(&mut self, seg_bytes: i64) {
        self.seg_req = if seg_bytes > 0 { seg_bytes } else { 0 };
    }
    pub fn segmented(&self) -> bool {
        !self.segs.is_empty()
    }
    pub fn segment_bytes(&self) -> i64 {
        self.seg
    }

    pub fn slots(&self) -> i64 {
        self.live_slots
    }
    pub fn full_slots(&self) -> i64 {
        self.allocated
    }
    /// Slots actually claimed. Not the same as `slots()` - the cache does not evict,
    /// so a run that routes fewer distinct experts than there are slots leaves the
    /// rest empty.
    pub fn resident(&self) -> i64 {
        if self.per_layer {
            self.admitted
        } else {
            self.next_free
        }
    }
    fn slot_end(&self, n: i64) -> i64 {
        if self.off.is_empty() {
            n * self.blob
        } else {
            self.off[n as usize] as i64
        }
    }
    pub fn bytes(&self) -> i64 {
        self.slot_end(self.live_slots)
    }
    pub fn full_bytes(&self) -> i64 {
        self.slot_end(self.allocated)
    }
    pub fn gib(&self) -> f64 {
        gib(self.bytes() as u64)
    }
    pub fn fills(&self) -> i64 {
        self.fills
    }
    pub fn set_per_layer_admission(&mut self, on: bool) {
        self.per_layer = on;
    }
    pub fn per_layer_admission(&self) -> bool {
        self.per_layer
    }

    /// Bytes of the arena backed by VRAM now (a whole number of segments; the arena
    /// when not segmented).
    pub fn mapped_bytes(&self) -> i64 {
        if self.segs.is_empty() {
            return if self.base != 0 { self.full_bytes() } else { 0 };
        }
        self.seg_size[..self.mapped_segs as usize].iter().sum()
    }

    /// The number of leading slots that fit wholly inside the first `bytes` bytes.
    pub fn slots_within(&self, bytes: i64) -> i64 {
        if bytes >= self.full_bytes() {
            return self.allocated;
        }
        if bytes <= 0 {
            return 0;
        }
        if self.off.is_empty() {
            return if self.blob > 0 { bytes / self.blob } else { 0 };
        }
        // off[i + 1] is slot i's end: count the slots whose end is at most `bytes`
        let tail = &self.off[1..];
        tail.partition_point(|&o| o <= bytes as u64) as i64
    }

    /// The bytes the first `n` slots span.
    pub fn bytes_of(&self, n: i64) -> i64 {
        self.slot_end(if n < 0 {
            0
        } else if n > self.allocated {
            self.allocated
        } else {
            n
        })
    }

    /// Allocates `n_slots * blob_bytes` of device memory and a
    /// `n_layers * n_expert` residency table, and checks the allocation against what
    /// the card actually has rather than assuming the planner was right.
    pub fn open(
        &mut self,
        n_slots: i64,
        n_layers: i64,
        n_expert: i64,
        blob_bytes: i64,
    ) -> Result<(), String> {
        self.close();
        if n_slots <= 0 {
            return Err("ExpertCache: n_slots must be positive".into());
        }
        if n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0 {
            return Err(
                "ExpertCache: n_layers, n_expert and blob_bytes must all be positive".into(),
            );
        }

        let want = (n_slots as u64) * (blob_bytes as u64);

        // The allocation is checked against the card, not against the request: a
        // machine where the weights already own most of VRAM would otherwise get a
        // cache that reports the slots it was ASKED for while device_slot() walks off
        // the end.
        if let Some((free, total)) = self.dev.mem_get_info() {
            if free < want {
                return Err(format!(
                    "ExpertCache: {n_slots} slots x {blob_bytes} B = {:.2} GiB, but only {:.2} GiB of VRAM is \
                     free ({:.2} GiB of {:.2} GiB total). Lower --expert-cache.",
                    gib(want),
                    gib(free),
                    gib(total - free),
                    gib(total)
                ));
            }
        }

        if self.seg_req > 0 {
            self.open_segmented(want)?;
        } else {
            self.base = self.dev.alloc(want).map_err(|e| {
                format!("ExpertCache: cudaMalloc({:.2} GiB) failed: {e}", gib(want))
            })?;
        }
        // Zeroed so a slot read before it is filled is a DETERMINISTIC wrong answer
        // rather than whatever the allocator handed back.
        if self.dev.memset(self.base, 0, want).is_err() {
            self.close();
            return Err("ExpertCache: cudaMemset of the slot arena failed".into());
        }

        self.residency = vec![NOT_RESIDENT; (n_layers * n_expert) as usize];
        self.allocated = n_slots;
        self.live_slots = n_slots;
        self.n_layers = n_layers;
        self.n_expert = n_expert;
        self.blob = blob_bytes;
        self.next_free = 0;
        self.fills = 0;
        self.admitted = 0;
        // Each layer starts at the bottom of its own range. Built here rather than
        // lazily so `admit` stays allocation-free on the token path.
        self.layer_next = vec![
            0;
            if self.n_layers > 0 {
                self.n_layers as usize
            } else {
                0
            }
        ];
        for l in 0..self.n_layers {
            let (lo, _hi) = self.layer_slot_range(l);
            self.layer_next[l as usize] = lo as i32;
        }
        Ok(())
    }

    /// Plan v0.3 P6: slots of the given sizes, back to back (a native pack's blobs
    /// differ per layer, and a profile-filled tier never moves an expert to another
    /// layer's slot, so each slot keeps its first size).
    pub fn open_sized(
        &mut self,
        slot_bytes: &[i64],
        n_layers: i64,
        n_expert: i64,
    ) -> Result<(), String> {
        if slot_bytes.is_empty() {
            return Err("ExpertCache: no slots".into());
        }
        let mut mx = 0i64;
        let mut off = vec![0u64; slot_bytes.len() + 1];
        for i in 0..slot_bytes.len() {
            // 256-byte aligned slots, so every blob starts where the kernels' vector loads expect it
            off[i + 1] = off[i] + (slot_bytes[i] as u64).div_ceil(256) * 256;
            if slot_bytes[i] > mx {
                mx = slot_bytes[i];
            }
        }
        // one allocation of the summed size, through the uniform path's checks: n "slots" of 1 byte
        self.open(*off.last().unwrap_or(&0) as i64, n_layers, n_expert, 1)?;
        self.allocated = slot_bytes.len() as i64;
        self.live_slots = self.allocated;
        self.blob = mx;
        self.off = off;
        // Each layer's cursor at the bottom of its own range, as open() seeds it -
        // open() above ran on byte-sized "slots", so its seeds are not slot indices.
        self.layer_next = vec![
            0;
            if self.n_layers > 0 {
                self.n_layers as usize
            } else {
                0
            }
        ];
        for l in 0..self.n_layers {
            let (lo, _hi) = self.layer_slot_range(l);
            self.layer_next[l as usize] = lo as i32;
        }
        Ok(())
    }

    fn open_segmented(&mut self, want: u64) -> Result<(), String> {
        if !self.dev.vmm_supported() {
            return Err("ExpertCache: --vram-elastic needs the driver's virtual memory management, which this GPU \
                        or driver does not offer"
                .into());
        }
        let g = self.dev.granularity().ok_or_else(|| {
            "ExpertCache: cannot read the driver's allocation granularity".to_string()
        })?;
        let total = want.div_ceil(g) * g;
        self.seg = (self.seg_req as u64).div_ceil(g) as i64 * g as i64;
        self.base = self.dev.address_reserve(total).ok_or_else(|| {
            "ExpertCache: cannot reserve the address range of the segmented expert cache"
                .to_string()
        })?;
        self.reserved = total;
        let mut at = 0u64;
        while at < total {
            self.segs.push(0);
            self.seg_size
                .push(std::cmp::min(self.seg as u64, total - at) as i64);
            at += self.seg as u64;
        }
        for i in 0..self.segs.len() {
            if !self.map_segment(
                self.base + (i as u64) * (self.seg as u64),
                self.seg_size[i] as u64,
                i,
            ) {
                return Err(format!(
                    "ExpertCache: cudaMalloc failed: segment {} of {} ({:.2} GiB) of the segmented cache could \
                     not be allocated",
                    i + 1,
                    self.segs.len(),
                    gib(self.seg_size[i] as u64)
                ));
            }
            self.mapped_segs = i as i64 + 1;
        }
        Ok(())
    }

    /// One segment: a physical allocation mapped at `addr`, readable and writable by
    /// this device. Each step rolls back the ones before it on failure.
    fn map_segment(&mut self, addr: u64, bytes: u64, index: usize) -> bool {
        let Some(handle) = self.dev.mem_create(bytes) else {
            return false;
        };
        if !self.dev.mem_map(addr, bytes, handle) {
            self.dev.mem_release(handle);
            return false;
        }
        if !self.dev.mem_set_access(addr, bytes) {
            self.dev.mem_unmap(addr, bytes);
            self.dev.mem_release(handle);
            return false;
        }
        self.segs[index] = handle;
        true
    }

    fn release_segmented(&mut self) {
        if self.base != 0 {
            let _ = self.dev.sync_device();
        }
        for i in 0..self.segs.len() {
            if self.segs[i] != 0 {
                let addr = self.base + (i as u64) * (self.seg as u64);
                let bytes = self.seg_size[i] as u64;
                self.dev.mem_unmap(addr, bytes);
                self.dev.mem_release(self.segs[i]);
            }
        }
        if self.base != 0 && self.reserved > 0 {
            self.dev.address_free(self.base, self.reserved);
        }
        self.segs.clear();
        self.seg_size.clear();
        self.mapped_segs = 0;
        self.reserved = 0;
        self.base = 0;
    }

    /// Unmaps every segment past the first `keep_bytes` (rounded UP to a segment
    /// boundary): `slots()` becomes the slots wholly inside what stays. Waits for the
    /// device first.
    pub fn shrink(&mut self, keep_bytes: i64) -> Result<(), String> {
        if self.segs.is_empty() {
            return Err(
                "the expert cache is not segmented (the engine needs --vram-elastic)".into(),
            );
        }
        // the segments [0, keep) hold the first keep_bytes
        let (mut keep, mut at) = (0i64, 0i64);
        while keep < self.segs.len() as i64 && at < keep_bytes {
            at += self.seg_size[keep as usize];
            keep += 1;
        }
        if keep >= self.mapped_segs {
            return Ok(());
        }
        self.dev
            .sync_device()
            .map_err(|e| format!("the device failed before the cache shrank: {e}"))?;
        let mut i = self.mapped_segs - 1;
        loop {
            let addr = self.base + (i as u64) * (self.seg as u64);
            let bytes = self.seg_size[i as usize] as u64;
            if !self.dev.mem_unmap(addr, bytes) || !self.dev.mem_release(self.segs[i as usize]) {
                self.live_slots = self.slots_within(self.mapped_bytes());
                return Err("the driver refused to release an expert-cache segment".into());
            }
            self.segs[i as usize] = 0;
            self.mapped_segs = i;
            if i == keep {
                break;
            }
            i -= 1;
        }
        self.live_slots = self.slots_within(self.mapped_bytes());
        Ok(())
    }

    /// Maps segments again up to `want_bytes` (rounded DOWN to a segment, at most the
    /// arena; the last, shorter segment only when `want_bytes` covers the arena).
    /// Stops at the first segment the driver cannot back. The new slots are empty
    /// until the caller fills them.
    pub fn grow(&mut self, want_bytes: i64) -> Result<(), String> {
        if self.segs.is_empty() {
            return Err(
                "the expert cache is not segmented (the engine needs --vram-elastic)".into(),
            );
        }
        let mut at = self.mapped_bytes();
        while self.mapped_segs < self.segs.len() as i64
            && at + self.seg_size[self.mapped_segs as usize] <= want_bytes
        {
            let i = self.mapped_segs as usize;
            if !self.map_segment(
                self.base + (i as u64) * (self.seg as u64),
                self.seg_size[i] as u64,
                i,
            ) {
                self.live_slots = self.slots_within(self.mapped_bytes());
                return Err("the driver has no VRAM for another expert-cache segment".into());
            }
            at += self.seg_size[i];
            self.mapped_segs += 1;
        }
        self.live_slots = self.slots_within(self.mapped_bytes());
        Ok(())
    }

    pub fn close(&mut self) {
        self.off.clear();
        if !self.segs.is_empty() {
            self.release_segmented();
        } else if self.base != 0 {
            self.dev.free(self.base);
            self.base = 0;
        }
        self.residency.clear();
        self.allocated = 0;
        self.live_slots = 0;
        self.n_layers = 0;
        self.n_expert = 0;
        self.blob = 0;
        self.next_free = 0;
        self.fills = 0;
        self.admitted = 0;
        self.layer_next.clear();
    }

    /// The slot range layer `l` may admit into under per-layer admission. Layer `l`
    /// owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the LAST layer takes
    /// whatever is left over, so the ranges always cover `0..slots` exactly.
    pub fn layer_slot_range(&self, layer: i64) -> (i64, i64) {
        let (mut lo, mut hi) = (0i64, 0i64);
        if self.n_layers <= 0 || self.allocated <= 0 || layer < 0 || layer >= self.n_layers {
            return (lo, hi);
        }
        let q = self.allocated / self.n_layers;
        lo = layer * q;
        hi = if layer == self.n_layers - 1 {
            self.allocated
        } else {
            (layer + 1) * q
        };
        (lo, hi)
    }

    /// `(layer, expert)` -> slot index, or `NOT_RESIDENT`. Bounds-checked: a bad layer
    /// or expert returns `NOT_RESIDENT` rather than reading whatever is adjacent in
    /// the table.
    pub fn slot_of(&self, layer: i64, expert: i64) -> i32 {
        if layer < 0 || layer >= self.n_layers || expert < 0 || expert >= self.n_expert {
            return NOT_RESIDENT;
        }
        self.residency[(layer * self.n_expert + expert) as usize]
    }

    /// Claims the next free slot for `(layer, expert)`. Returns the slot, or
    /// `NOT_RESIDENT` when the cache is full - it never evicts.
    pub fn admit(&mut self, layer: i64, expert: i64) -> i32 {
        if layer < 0 || layer >= self.n_layers || expert < 0 || expert >= self.n_expert {
            return NOT_RESIDENT;
        }
        let at = (layer * self.n_expert + expert) as usize;
        if self.residency[at] != NOT_RESIDENT {
            return self.residency[at];
        }
        if self.per_layer {
            if self.layer_next.is_empty() {
                return NOT_RESIDENT;
            }
            let (_lo, hi) = self.layer_slot_range(layer);
            if (self.layer_next[layer as usize] as i64) >= hi {
                return NOT_RESIDENT; // this layer's quota is full
            }
            self.residency[at] = self.layer_next[layer as usize];
            self.layer_next[layer as usize] += 1;
            self.admitted += 1;
            return self.residency[at];
        }
        if self.next_free >= self.allocated {
            return NOT_RESIDENT; // full: no eviction, deliberately
        }
        self.residency[at] = self.next_free as i32;
        self.next_free += 1;
        self.residency[at]
    }

    /// Publish a same-layer replacement after its slot copy has completed.
    ///
    /// A self-replace CLEARS the entry: the C++ reads through a reference into the
    /// array, so when both indices land on the same element the second write lands
    /// there too. Kept exactly, because a caller that ever passes the same expert
    /// twice would see a different result from one that does not.
    pub fn replace(&mut self, layer: i64, old_expert: i32, new_expert: i32) {
        let old = (layer * self.n_expert + i64::from(old_expert)) as usize;
        let new = (layer * self.n_expert + i64::from(new_expert)) as usize;
        let slot = self.residency[old];
        self.residency[new] = slot;
        self.residency[old] = NOT_RESIDENT;
    }

    /// The device address of one slot.
    pub fn device_slot(&self, slot: i32) -> Option<u64> {
        if slot < 0 || i64::from(slot) >= self.allocated {
            return None;
        }
        let base = if self.off.is_empty() {
            self.base + (i64::from(slot) as u64) * (self.blob as u64)
        } else {
            self.base + self.off[slot as usize]
        };
        Some(base)
    }

    /// Copies the blob into `slot`. Asynchronous: the caller orders it.
    pub fn fill_slot(
        &mut self,
        slot: i32,
        host_blob: &[u8],
        stream: u64,
        bytes: i64,
    ) -> Result<(), String> {
        let n = if bytes > 0 && bytes <= self.blob {
            bytes
        } else {
            self.blob
        };
        let dst = match self.device_slot(slot) {
            Some(d) => d,
            None => {
                return Err(format!(
                    "ExpertCache::fill_slot: slot {slot} is outside 0..{}",
                    self.allocated - 1
                ))
            }
        };
        if host_blob.is_empty() {
            return Err("ExpertCache::fill_slot: the host blob is null".into());
        }
        self.dev
            .memcpy_h2d_async(dst, &host_blob[..n as usize], stream)
            .map_err(|e| format!("ExpertCache::fill_slot: {e}"))?;
        self.fills += 1;
        Ok(())
    }

    /// The same copy, but blocking, and the profile fill needs it: at startup there is
    /// no stream ordering to lean on - the source is pageable host memory,
    /// `verify_slot` reads the slot back on the LEGACY stream, and a legacy-stream
    /// copy is not ordered against a non-blocking one.
    pub fn fill_slot_blocking(
        &mut self,
        slot: i32,
        host_blob: &[u8],
        bytes: i64,
    ) -> Result<(), String> {
        let n = if bytes > 0 && bytes <= self.blob {
            bytes
        } else {
            self.blob
        };
        let dst = match self.device_slot(slot) {
            Some(d) => d,
            None => return Err("ExpertCache::fill_slot_blocking: slot outside the arena".into()),
        };
        if host_blob.is_empty() {
            return Err("ExpertCache::fill_slot_blocking: the host blob is null".into());
        }
        self.dev
            .memcpy_h2d_sync(dst, &host_blob[..n as usize])
            .map_err(|e| format!("ExpertCache::fill_slot_blocking: {e}"))?;
        self.fills += 1;
        Ok(())
    }

    /// perf-review D-4: the blocking form's copy on the same (legacy) stream, but
    /// queued: many slots are refilled with one `sync_queued` at the end instead of a
    /// wait per slot. Same ordering against earlier work, same bytes.
    pub fn fill_slot_queued(
        &mut self,
        slot: i32,
        host_blob: &[u8],
        bytes: i64,
    ) -> Result<(), String> {
        let n = if bytes > 0 && bytes <= self.blob {
            bytes
        } else {
            self.blob
        };
        let dst = match self.device_slot(slot) {
            Some(d) => d,
            None => return Err("ExpertCache::fill_slot_queued: slot outside the arena".into()),
        };
        if host_blob.is_empty() {
            return Err("ExpertCache::fill_slot_queued: the host blob is null".into());
        }
        self.dev
            .memcpy_h2d_async(dst, &host_blob[..n as usize], 0)
            .map_err(|e| format!("ExpertCache::fill_slot_queued: {e}"))?;
        self.fills += 1;
        Ok(())
    }

    pub fn sync_queued(&mut self, stream: u64) -> Result<(), String> {
        self.dev
            .sync_stream(stream)
            .map_err(|e| format!("ExpertCache::sync_queued: {e}"))
    }

    /// Reads `slot` back and compares it byte for byte. The only thing that says the
    /// cache holds the expert it claims to: a slot table that is right about indices
    /// and wrong about bytes produces a plausible token.
    pub fn verify_slot(&mut self, slot: i32, host_blob: &[u8], bytes: i64) -> Result<(), String> {
        let n = if bytes > 0 && bytes <= self.blob {
            bytes
        } else {
            self.blob
        };
        let src = match self.device_slot(slot) {
            Some(s) => s,
            None => return Err("ExpertCache::verify_slot: slot outside the arena".into()),
        };
        let mut got = vec![0u8; n as usize];
        self.dev
            .memcpy_d2h(&mut got, src)
            .map_err(|e| format!("ExpertCache::verify_slot: {e}"))?;
        if got[..] != host_blob[..n as usize] {
            let first = got[..n as usize]
                .iter()
                .zip(host_blob[..n as usize].iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return Err(format!(
                "ExpertCache::verify_slot: slot {slot} differs from the arena at byte {first} (of {})",
                self.blob
            ));
        }
        Ok(())
    }
}
