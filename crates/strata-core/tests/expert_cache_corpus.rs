// Replay of tools/expert_cache_corpus.cpp: the same scenarios, the same fake device
// rules, the same output. See the harness header for what is covered and what is not.

use std::collections::HashMap;
use strata_core::expert_cache::{
    rank_learned_profile, read_expert_profile, write_expert_profile, Device, ExpertCache,
    NOT_RESIDENT,
};

const ERR_NO_ERROR: &str = "no error";
const ERR_OOM: &str = "out of memory";

struct Fake {
    alloc_fail: bool,
    memset_fail: bool,
    info_ok: bool,
    free_b: u64,
    total_b: u64,
    sync_ok: bool,
    stream_sync_ok: bool,
    fail_copy_bytes: usize,
    vmm_supported: bool,
    gran_ok: bool,
    reserve_ok: bool,
    gran: u64,
    fail_create_at: i32,
    fail_map_at: i32,
    fail_access_at: i32,
    // bookkeeping: device memory is ONE arena, so any address within a reservation
    // resolves - which is what makes a segment at an offset inside the range work.
    arena: Vec<u8>,
    arena_next: usize,
    handles: HashMap<u64, Vec<u8>>,
    maps: HashMap<u64, u64>,
    next_handle: u64,
    create_n: i32,
    map_n: i32,
    access_n: i32,
}

impl Default for Fake {
    fn default() -> Self {
        Fake {
            alloc_fail: false,
            memset_fail: false,
            info_ok: true,
            free_b: 64 << 30,
            total_b: 64 << 30,
            sync_ok: true,
            stream_sync_ok: true,
            fail_copy_bytes: 0,
            vmm_supported: true,
            gran_ok: true,
            reserve_ok: true,
            gran: 2 << 20,
            fail_create_at: -1,
            fail_map_at: -1,
            fail_access_at: -1,
            arena: vec![0u8; ARENA],
            arena_next: 0,
            handles: HashMap::new(),
            maps: HashMap::new(),
            next_handle: 1,
            create_n: 0,
            map_n: 0,
            access_n: 0,
        }
    }
}

const ARENA: usize = 256 << 20;
const BASE: u64 = 0x1000_0000;

impl Fake {
    fn alloc_zeroed(&mut self, n: usize) -> Option<u64> {
        if n > ARENA - self.arena_next {
            return None;
        }
        let at = self.arena_next;
        self.arena_next += n;
        self.arena[at..at + n].fill(0);
        Some(BASE + at as u64)
    }
    fn slot(&mut self, addr: u64, n: usize) -> Option<&mut [u8]> {
        let at = (addr - BASE) as usize;
        if at + n > ARENA {
            return None;
        }
        Some(&mut self.arena[at..at + n])
    }
}

impl Device for Fake {
    fn mem_get_info(&mut self) -> Option<(u64, u64)> {
        if self.info_ok {
            Some((self.free_b, self.total_b))
        } else {
            None
        }
    }
    fn alloc(&mut self, bytes: u64) -> Result<u64, String> {
        if self.alloc_fail {
            return Err(ERR_OOM.to_string()); // the real runtime leaves the error set for the caller to read
        }
        self.alloc_zeroed(bytes as usize)
            .ok_or_else(|| ERR_OOM.to_string())
    }
    fn free(&mut self, _addr: u64) {}
    fn memset(&mut self, addr: u64, value: u8, bytes: u64) -> Result<(), String> {
        if self.memset_fail {
            return Err(ERR_NO_ERROR.to_string());
        }
        if let Some(b) = self.slot(addr, bytes as usize) {
            b.fill(value);
        }
        Ok(())
    }
    fn memcpy_h2d_async(&mut self, dst: u64, src: &[u8], _stream: u64) -> Result<(), String> {
        self.memcpy_h2d_sync(dst, src)
    }
    fn memcpy_h2d_sync(&mut self, dst: u64, src: &[u8]) -> Result<(), String> {
        if src.len() == self.fail_copy_bytes {
            return Err(ERR_OOM.to_string());
        }
        // device memory is host memory here; a copy is a copy
        let n = src.len();
        let Some(entry) = self.slot(dst, n) else {
            return Err(ERR_NO_ERROR.to_string());
        };
        entry.copy_from_slice(&src[..n]);
        Ok(())
    }
    fn memcpy_d2h(&mut self, dst: &mut [u8], src: u64) -> Result<(), String> {
        if dst.len() == self.fail_copy_bytes {
            return Err(ERR_OOM.to_string());
        }
        let n = dst.len();
        let at = (src - BASE) as usize;
        if at + n > ARENA {
            return Err(ERR_NO_ERROR.to_string());
        }
        dst.copy_from_slice(&self.arena[at..at + n]);
        Ok(())
    }
    fn sync_device(&mut self) -> Result<(), String> {
        if self.sync_ok {
            Ok(())
        } else {
            Err(ERR_OOM.to_string())
        }
    }
    fn sync_stream(&mut self, _stream: u64) -> Result<(), String> {
        if self.stream_sync_ok {
            Ok(())
        } else {
            Err(ERR_OOM.to_string())
        }
    }
    fn vmm_supported(&mut self) -> bool {
        self.vmm_supported
    }
    fn granularity(&mut self) -> Option<u64> {
        if self.gran_ok {
            Some(self.gran)
        } else {
            None
        }
    }
    fn address_reserve(&mut self, bytes: u64) -> Option<u64> {
        if !self.reserve_ok {
            return None;
        }
        self.alloc_zeroed(bytes as usize)
    }
    fn address_free(&mut self, _addr: u64, _bytes: u64) {}
    fn mem_create(&mut self, bytes: u64) -> Option<u64> {
        self.create_n += 1;
        if self.create_n == self.fail_create_at {
            return None;
        }
        let buf = vec![0u8; bytes as usize];
        let h = self.next_handle;
        self.next_handle += 1;
        self.handles.insert(h, buf);
        Some(h)
    }
    fn mem_release(&mut self, handle: u64) -> bool {
        self.handles.remove(&handle);
        true
    }
    fn mem_map(&mut self, addr: u64, bytes: u64, handle: u64) -> bool {
        self.map_n += 1;
        if self.map_n == self.fail_map_at {
            return false;
        }
        let Some(handle_buf) = self.handles.get(&handle).cloned() else {
            return false;
        };
        let Some(range) = self.slot(addr, bytes as usize) else {
            return false;
        };
        // the physical's bytes land in the range
        range.copy_from_slice(&handle_buf[..bytes as usize]);
        self.maps.insert(addr, handle);
        true
    }
    fn mem_unmap(&mut self, addr: u64, bytes: u64) -> bool {
        if let Some(h) = self.maps.remove(&addr) {
            if let Some(range) = self.slot(addr, bytes as usize).map(|r| r.to_vec()) {
                if let Some(handle_buf) = self.handles.get_mut(&h) {
                    // and go back the same way
                    handle_buf[..bytes as usize].copy_from_slice(&range[..bytes as usize]);
                }
            }
        }
        true
    }
    fn mem_set_access(&mut self, _addr: u64, _bytes: u64) -> bool {
        self.access_n += 1;
        self.access_n != self.fail_access_at
    }
}

// ---- the golden

struct Replay {
    out: Vec<String>,
}

impl Replay {
    fn new() -> Self {
        Replay { out: Vec::new() }
    }
    fn line(&mut self, s: String) {
        self.out.push(format!("CORPUS|{s}"));
    }
}

fn s16(v: &[(i32, i32)], n: usize) -> String {
    let shown: Vec<String> = v.iter().take(n).map(|(l, e)| format!("{l}:{e}")).collect();
    let mut s = shown.join(",");
    if v.len() > n {
        s += ",...";
    }
    s
}

const K_PATH: &str = "/tmp/strata_ec_corpus_profile.bin";

fn write_file(b: &[u8], path: &str) {
    std::fs::write(path, b).unwrap();
}

fn make_profile(
    nl: u32,
    ne: u32,
    slots: u32,
    ranked: &[(u16, u16)],
    table: &[i32],
    magic: &[u8; 4],
) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(magic);
    for v in [1u32, nl, ne, slots, ranked.len() as u32] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for (l, e) in ranked {
        b.extend_from_slice(&l.to_le_bytes());
        b.extend_from_slice(&e.to_le_bytes());
    }
    for t in table {
        b.extend_from_slice(&t.to_le_bytes());
    }
    b
}

fn scenario_profile(r: &mut Replay) {
    // a good file: 2 layers x 3 experts, 4 slots, 2 ranked
    write_file(
        &make_profile(2, 3, 4, &[(1, 2), (0, 0)], &[-1, -1, -1, 1, -1, 0], b"STRP"),
        K_PATH,
    );
    match read_expert_profile(K_PATH, 2, 3) {
        Ok((ranked, slots)) => r.line(format!("PROFILE|read|{}|{slots}", s16(&ranked, 8))),
        Err(e) => r.line(format!("PROFILE|read|FAIL|{e}")),
    }

    write_file(&make_profile(2, 3, 4, &[], &[], b"XSTR"), K_PATH);
    if let Err(e) = read_expert_profile(K_PATH, 2, 3) {
        r.line(format!("PROFILE|magic|{e}"));
    }

    // too short to hold even a header
    write_file(b"STRP\x01\x00\x00\x00", K_PATH);
    if let Err(e) = read_expert_profile(K_PATH, 2, 3) {
        r.line(format!("PROFILE|short|{e}"));
    }

    write_file(&make_profile(3, 3, 4, &[], &[], b"STRP"), K_PATH);
    if let Err(e) = read_expert_profile(K_PATH, 2, 3) {
        r.line(format!("PROFILE|dims|{e}"));
    }

    write_file(
        &make_profile(2, 3, 1, &[(0, 0), (1, 2)], &[], b"STRP"),
        K_PATH,
    );
    if let Err(e) = read_expert_profile(K_PATH, 2, 3) {
        r.line(format!("PROFILE|toomany|{e}"));
    }

    // header says 2 pairs, only one present
    let mut trunc = make_profile(2, 3, 4, &[(0, 0), (1, 2)], &[], b"STRP");
    trunc.truncate(trunc.len() - 2);
    write_file(&trunc, K_PATH);
    if let Err(e) = read_expert_profile(K_PATH, 2, 3) {
        r.line(format!("PROFILE|truncated|{e}"));
    }

    write_file(
        &make_profile(2, 3, 4, &[(0, 0), (5, 2)], &[], b"STRP"),
        K_PATH,
    );
    if let Err(e) = read_expert_profile(K_PATH, 2, 3) {
        r.line(format!("PROFILE|range|{e}"));
    }

    write_file(&make_profile(2, 3, 4, &[], &[], b"STRP"), K_PATH);
    if let Ok((ranked, slots)) = read_expert_profile(K_PATH, 2, 3) {
        r.line(format!("PROFILE|empty|{}|{slots}", ranked.len()));
    }

    // the writer, and what it refuses
    let input = vec![(1, 2), (0, 0)];
    if write_expert_profile(K_PATH, 2, 3, &input).is_ok() {
        let got = std::fs::read(K_PATH).unwrap();
        let hex: String = got[..28.min(got.len())]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        r.line(format!("PROFILE|write|{}|{hex}", got.len()));
        // and read it back through the real reader
        match read_expert_profile(K_PATH, 2, 3) {
            Ok((back, bs)) => r.line(format!("PROFILE|roundtrip|{}|{bs}", s16(&back, 8))),
            Err(e) => r.line(format!("PROFILE|roundtrip|FAIL|{e}")),
        }
    } else {
        r.line("PROFILE|write|FAIL".to_string());
    }
    if let Err(e) = write_expert_profile(K_PATH, 2, 3, &[(7, 0)]) {
        r.line(format!("PROFILE|write_range|{e}"));
    }
    if let Err(e) = write_expert_profile(K_PATH, 0, 3, &input) {
        r.line(format!("PROFILE|write_fit|{e}"));
    }
    let _ = std::fs::remove_file(K_PATH);
    let _ = std::fs::remove_file(format!("{K_PATH}.tmp"));
}

fn scenario_rank(r: &mut Replay) {
    // 2 x 3. resident: 0,1 and 1,2. heat: 5 at 0,2 and 1 at 1,0. prior: 1,1 then 0,0.
    let res = [0u8, 1, 0, 0, 0, 1];
    let heat = [0f64, 0.0, 5.0, 1.0, 0.0, 0.0];
    let prior = [(1, 1), (0, 0), (9, 9), (1, 1)];
    r.line(format!(
        "RANK|both|{}",
        s16(&rank_learned_profile(2, 3, &res, &heat, &prior), 6)
    ));
    r.line(format!(
        "RANK|noheat|{}",
        s16(&rank_learned_profile(2, 3, &res, &[], &prior), 6)
    ));
    r.line(format!(
        "RANK|noprior|{}",
        s16(&rank_learned_profile(2, 3, &res, &heat, &[]), 6)
    ));
    r.line(format!(
        "RANK|empty|{}",
        s16(&rank_learned_profile(2, 3, &[], &[], &[]), 6)
    ));
    // all equal: index order decides
    let none = [0u8, 0, 0, 0, 0, 0];
    r.line(format!(
        "RANK|ties|{}",
        s16(&rank_learned_profile(2, 3, &none, &[], &[]), 6)
    ));
    // a short resident vector: past its end is "not resident", heat 0
    let shortres = [1u8, 0];
    r.line(format!(
        "RANK|shortvec|{}",
        s16(&rank_learned_profile(2, 3, &shortres, &[], &[]), 6)
    ));
}

fn line_va(r: &mut Replay, label: &str, c: &ExpertCache<Fake>) {
    r.line(format!(
        "{label}|slots={}|full={}|bytes={}|full_bytes={}|gib={:.4}|resident={}|seg={}|segbytes={}|mapped={}",
        c.slots(),
        c.full_slots(),
        c.bytes(),
        c.full_bytes(),
        c.gib(),
        c.resident(),
        if c.segmented() { 1 } else { 0 },
        c.segment_bytes(),
        c.mapped_bytes()
    ));
}

fn new_cache(fake: Fake) -> ExpertCache<Fake> {
    ExpertCache::new(fake)
}

fn scenario_open(r: &mut Replay) {
    let mut c = new_cache(Fake::default());
    let cases: [(&str, i64, i64, i64, i64); 5] = [
        ("slots0", 0, 48, 512, 1310720),
        ("slotsneg", -1, 48, 512, 1310720),
        ("layers0", 4096, 0, 512, 1310720),
        ("expert0", 4096, 48, 0, 1310720),
        ("blob0", 4096, 48, 512, 0),
    ];
    for (label, slots, nl, ne, blob) in cases {
        match c.open(slots, nl, ne, blob) {
            Ok(()) => r.line(format!("OPEN|refuse|{label}|1|")),
            Err(e) => r.line(format!("OPEN|refuse|{label}|0|{e}")),
        }
    }

    // the allocation is checked against the card before it is made
    let f = Fake {
        free_b: 4 << 30,
        total_b: 24 << 30,
        ..Fake::default()
    };
    let mut c = new_cache(f);
    if let Err(e) = c.open(8192, 48, 512, 1310720) {
        r.line(format!("OPEN|vram|{e}"));
    }

    let f = Fake {
        info_ok: false,
        alloc_fail: true,
        ..Fake::default()
    };
    let mut c = new_cache(f);
    if let Err(e) = c.open(8, 4, 8, 1024) {
        r.line(format!("OPEN|alloc|{e}"));
    }

    let f = Fake {
        info_ok: false,
        ..Fake::default()
    };
    let mut c = new_cache(f);
    if c.open(8, 4, 8, 1024).is_ok() {
        r.line("OPEN|infofail|the check is skipped and the allocation is made".to_string());
    }
    c.close();

    let f = Fake {
        info_ok: false,
        memset_fail: true,
        ..Fake::default()
    };
    let mut c = new_cache(f);
    if let Err(e) = c.open(8, 4, 8, 1024) {
        r.line(format!("OPEN|memset|{e}"));
    }

    let mut c = new_cache(Fake::default());
    if c.open(64, 48, 512, 1310720).is_ok() {
        line_va(r, "OPEN|ok", &c);
        r.line(format!(
            "OPEN|within|{}|{}|{}|{}",
            c.slots_within(1310720),
            c.slots_within(2621440),
            c.slots_within(83886080),
            c.slots_within(0)
        ));
        r.line(format!(
            "OPEN|bytesof|{}|{}|{}|{}",
            c.bytes_of(0),
            c.bytes_of(3),
            c.bytes_of(99),
            c.bytes_of(-1)
        ));
    } else {
        r.line("OPEN|ok|FAIL".to_string());
    }
    c.close();

    // sized slots: 256-byte aligned, each slot keeps its own size
    let mut c = new_cache(Fake::default());
    let sizes = [1000i64, 2000, 256, 0];
    if c.open_sized(&sizes, 2, 4).is_ok() {
        line_va(r, "SIZED|ok", &c);
        r.line(format!(
            "SIZED|within|{}|{}|{}",
            c.slots_within(1024),
            c.slots_within(2304),
            c.slots_within(4352)
        ));
        r.line(format!(
            "SIZED|bytesof|{}|{}|{}",
            c.bytes_of(0),
            c.bytes_of(2),
            c.bytes_of(4)
        ));
    } else {
        r.line("SIZED|ok|FAIL".to_string());
    }
    if let Err(e) = c.open_sized(&[], 2, 4) {
        r.line(format!("SIZED|refuse|{e}"));
    }
    c.close();
}

fn scenario_segmented(r: &mut Replay) {
    let f = Fake {
        gran: 4096,
        ..Fake::default()
    };
    let mut c = new_cache(f);
    c.set_segment_bytes(16384);
    if c.open(16, 2, 4, 4096).is_ok() {
        // the fake's granularity is 4096 here, as the harness sets it
        line_va(r, "SEG|open", &c);
        r.line(format!(
            "SEG|within|{}|{}|{}",
            c.slots_within(16384),
            c.slots_within(36864),
            c.slots_within(65536)
        ));
        // shrink rounds UP to a segment: keeping 20000 bytes keeps 2 segments
        match c.shrink(20000) {
            Ok(()) => line_va(r, "SEG|shrink20000", &c),
            Err(e) => r.line(format!("SEG|shrink20000|FAIL|{e}")),
        }
        // and grow rounds DOWN: wanting 36000 gets 2 segments, which is what it has
        match c.grow(36000) {
            Ok(()) => line_va(r, "SEG|grow36000", &c),
            Err(e) => r.line(format!("SEG|grow36000|FAIL|{e}")),
        }
        match c.grow(65536) {
            Ok(()) => line_va(r, "SEG|growall", &c),
            Err(e) => r.line(format!("SEG|growall|FAIL|{e}")),
        }
    } else {
        r.line("SEG|open|FAIL".to_string());
    }
    c.close();

    // close() does NOT reset the segment request, so this cache is still segmented and
    // the "not segmented" refusal below is unreachable for it - which is what the golden
    // shows, because the C++ reuses the object. A genuinely unsegmented cache refuses.
    line_va(r, "SEG|reopened", &c);
    let mut plain = new_cache(Fake::default());
    if plain.open(4, 2, 4, 1024).is_ok() {
        if let Err(e) = plain.shrink(1024) {
            r.line(format!("SEG|shrink_unseg|{e}"));
        }
        if let Err(e) = plain.grow(4096) {
            r.line(format!("SEG|grow_unseg|{e}"));
        }
    }
    plain.close();

    // the driver cannot back the third segment: the message names it, and what was mapped stays
    let f = Fake {
        gran: 4096,
        fail_create_at: 3,
        ..Fake::default()
    };
    let mut c = new_cache(f);
    c.set_segment_bytes(16384);
    if let Err(e) = c.open(16, 2, 4, 4096) {
        r.line(format!("SEG|createfail|{e}"));
    } else {
        r.line("SEG|createfail|unexpectedly ok".to_string());
    }
    c.close();

    let f = Fake {
        gran: 4096,
        vmm_supported: false,
        ..Fake::default()
    };
    let mut c = new_cache(f);
    c.set_segment_bytes(16384);
    if let Err(e) = c.open(16, 2, 4, 4096) {
        r.line(format!("SEG|nosupport|{e}"));
    }
    c.close();

    let f = Fake {
        gran: 4096,
        gran_ok: false,
        ..Fake::default()
    };
    let mut c = new_cache(f);
    c.set_segment_bytes(16384);
    if let Err(e) = c.open(16, 2, 4, 4096) {
        r.line(format!("SEG|granfail|{e}"));
    }
    c.close();

    // what stays mapped keeps its bytes; what is unmapped and mapped again comes back as fresh
    // physical memory, because the driver hands out a new allocation, not the old one.
    let f = Fake {
        gran: 4096,
        ..Fake::default()
    };
    let mut c = new_cache(f);
    c.set_segment_bytes(16384);
    if c.open(16, 2, 4, 4096).is_ok() {
        let host: Vec<u8> = (0..4096).map(|i| (i * 7 + 1) as u8).collect();
        let zeros = vec![0u8; 4096];
        let f2 = c.fill_slot(2, &host, 0, 0).is_ok(); // segment 0: the shrink keeps this one
        let f12 = c.fill_slot(12, &host, 0, 0).is_ok(); // segment 3: the shrink takes this one
        r.line(format!(
            "SEG|fill|{}|{}|{}",
            f2 as i32,
            f12 as i32,
            c.fills()
        ));
        let v2 = c.verify_slot(2, &host, 0).is_ok();
        let v12 = c.verify_slot(12, &host, 0).is_ok();
        r.line(format!("SEG|verify|{}|{}", v2 as i32, v12 as i32));
        // Nothing is read while a segment is unmapped - the address range has no backing then,
        // and what a read would return is the driver's business, not the cache's. What the corpus
        // pins is that after a regrow the segment is NEW physical memory, not the old bytes.
        if c.shrink(16384).is_ok() {
            // segments [0,1) stay
            line_va(r, "SEG|shrunk", &c);
            if c.grow(65536).is_ok() {
                line_va(r, "SEG|regrown", &c);
                let a = c.verify_slot(2, &host, 0).is_ok();
                let b = c.verify_slot(12, &zeros, 0).is_ok();
                r.line(format!("SEG|verifyregrown|{}|{}", a as i32, b as i32));
            }
        }
        c.close();
    }
}

fn scenario_admission(r: &mut Replay) {
    let mut c = new_cache(Fake::default());
    if c.open(6, 3, 4, 1024).is_err() {
        r.line("ADM|open failed".to_string());
        return;
    }

    // the global path: arrival order, no eviction
    let a = c.admit(0, 0);
    let b = c.admit(0, 1);
    let d2 = c.admit(1, 3);
    let e = c.admit(0, 0);
    let f = c.admit(2, 2);
    r.line(format!("ADM|global|{a}|{b}|{d2}|{e}|{f}"));
    let g = c.admit(2, 0);
    let h = c.admit(1, 0);
    let i = c.admit(1, 1);
    let j = c.admit(2, 3);
    r.line(format!("ADM|globalfill|{g}|{h}|{i}|{j}"));
    let k = c.admit(2, 1);
    let l = c.admit(0, 2);
    r.line(format!("ADM|globalfull|{k}|{l}"));
    r.line(format!("ADM|globalres|{}|{}", c.resident(), c.full_slots()));
    let m = c.admit(3, 0);
    let n = c.admit(0, 4);
    r.line(format!("ADM|globaloob|{m}|{n}|{}", c.slot_of(3, 0)));

    // a normal replace moves the slot; a self-replace CLEARS it, because the C++ reads
    // through a reference into the array and both indices land on the same element
    let o = c.slot_of(0, 0);
    c.replace(0, 0, 2);
    r.line(format!(
        "ADM|replace|{o}|{}|{}",
        c.slot_of(0, 2),
        c.slot_of(0, 0)
    ));
    let p = c.slot_of(0, 2);
    c.replace(0, 2, 2);
    r.line(format!("ADM|selfreplace|{p}|{}", c.slot_of(0, 2)));

    // per-layer: 6 slots over 3 layers is q=2, the last layer takes the remainder
    let mut d = new_cache(Fake::default());
    if d.open(7, 3, 4, 1024).is_err() {
        r.line("ADM|open2 failed".to_string());
        return;
    }
    d.set_per_layer_admission(true);
    for l in 0..3 {
        let (lo, hi) = d.layer_slot_range(l);
        r.line(format!("ADM|range|{l}|{lo}|{hi}"));
    }
    let (lo, hi) = d.layer_slot_range(9);
    r.line(format!("ADM|range_oob|{lo}|{hi}"));
    let a = d.admit(0, 0);
    let b = d.admit(0, 1);
    let e = d.admit(0, 2);
    r.line(format!("ADM|perlayer|{a}|{b}|{e}"));
    r.line(format!("ADM|perlayerfull|{}", d.admit(0, 3)));
    let f = d.admit(1, 0);
    let g = d.admit(2, 0);
    r.line(format!("ADM|otherlayer|{f}|{g}"));
    let h = d.admit(2, 1);
    let i = d.admit(2, 2);
    let j = d.admit(2, 3);
    r.line(format!("ADM|lastlayer|{h}|{i}|{j}"));
    r.line(format!(
        "ADM|perlayerres|{}|{}",
        d.resident(),
        d.full_slots()
    ));
    c.close();
    d.close();
}

fn scenario_copy(r: &mut Replay) {
    let mut c = new_cache(Fake::default());
    if c.open(4, 2, 4, 1024).is_err() {
        r.line("COPY|open failed".to_string());
        return;
    }
    let host: Vec<u8> = (0..1024).map(|i| (i * 31 + 5) as u8).collect();

    if c.fill_slot(0, &host, 0, 0).is_ok() {
        r.line(format!("COPY|fill0|{}", c.fills()));
    } else {
        r.line("COPY|fill0|FAIL".to_string());
    }
    if c.verify_slot(0, &host, 0).is_ok() {
        r.line("COPY|verify0|ok".to_string());
    } else {
        r.line("COPY|verify0|FAIL".to_string());
    }

    // a partial blob: bytes < the slot, so only that many are compared
    if c.fill_slot(1, &host, 0, 256).is_ok() {
        r.line(format!("COPY|fill1partial|{}", c.fills()));
    }
    let mut host2 = host.clone();
    host2[200] ^= 0xFF;
    if let Err(e) = c.verify_slot(1, &host2, 256) {
        r.line(format!("COPY|verifydiff|{e}"));
    }
    if c.verify_slot(1, &host, 0).is_ok() {
        r.line("COPY|verifyfull|ok".to_string()); // the byte the copy never wrote
    }

    // bytes larger than the slot falls back to the slot size
    if c.fill_slot(2, &host, 0, 9999).is_ok() {
        r.line(format!("COPY|fill2big|{}", c.fills()));
    }
    if let Err(e) = c.fill_slot(9, &host, 0, 0) {
        r.line(format!("COPY|oob|{e}"));
    }
    if let Err(e) = c.fill_slot(3, &[], 0, 0) {
        r.line(format!("COPY|null|{e}"));
    }

    // the blocking and queued forms, and their own messages
    if c.fill_slot_blocking(3, &host, 0).is_ok() {
        r.line(format!("COPY|block3|{}", c.fills()));
    } else {
        r.line("COPY|block3|FAIL".to_string());
    }
    if let Err(e) = c.fill_slot_blocking(9, &host, 0) {
        r.line(format!("COPY|blockoob|{e}"));
    }
    if let Err(e) = c.fill_slot_queued(9, &host, 0) {
        r.line(format!("COPY|queueoob|{e}"));
    }
    if let Err(e) = c.fill_slot_queued(3, &[], 0) {
        r.line(format!("COPY|queuenull|{e}"));
    }

    // a copy that fails says so, with the runtime's own words
    let f = Fake {
        fail_copy_bytes: 1024,
        ..Fake::default()
    };
    let mut c = new_cache(f);
    if c.open(4, 2, 4, 1024).is_ok() {
        if let Err(e) = c.fill_slot(0, &host, 0, 0) {
            r.line(format!("COPY|copyfail|{e}"));
        }
        if c.sync_queued(0).is_ok() {
            r.line("COPY|syncok|a failed copy does not poison the queue".to_string());
        }
        c.device_mut().stream_sync_ok = false;
        if let Err(e) = c.sync_queued(0) {
            r.line(format!("COPY|syncfail|{e}"));
        }
    }

    // verify reads back what it is told to compare, so a wrong slot is caught
    let mut c = new_cache(Fake::default());
    if c.open(4, 2, 4, 1024).is_ok() {
        let mut other = host.clone();
        other[7] = 0;
        if let Err(e) = c.verify_slot(0, &other, 0) {
            r.line(format!("COPY|verifyzero|{e}"));
        }
        r.line(format!(
            "COPY|slotbounds|{}|{}|{}",
            c.device_slot(0).is_some() as i32,
            c.device_slot(4).is_some() as i32,
            c.device_slot(-1).is_some() as i32
        ));
    }
    c.close();
}

fn scenario_close(r: &mut Replay) {
    let mut c = new_cache(Fake::default());
    if c.open(6, 3, 4, 1024).is_err() {
        r.line("CLOSE|open failed".to_string());
        return;
    }
    let host: Vec<u8> = (0..1024).map(|i| (i * 31 + 5) as u8).collect();
    c.set_per_layer_admission(true);
    c.admit(0, 0);
    let _ = c.fill_slot(0, &[], 0, 0); // fails: null host, no counter moved
    let _ = c.fill_slot(1, &host, 0, 0);
    r.line(format!(
        "CLOSE|before|{}|{}|{}",
        c.resident(),
        c.fills(),
        c.slot_of(0, 0)
    ));
    c.close();
    r.line(format!(
        "CLOSE|after|{}|{}|{}|{}|{}|{}",
        c.slots(),
        c.full_slots(),
        c.resident(),
        c.fills(),
        c.slot_of(0, 0),
        c.segment_bytes()
    ));
    // and it is reusable
    c.set_segment_bytes(0);
    c.set_per_layer_admission(false);
    if c.open(2, 1, 2, 512).is_ok() {
        r.line(format!("CLOSE|reopen|{}|{}", c.bytes(), c.resident()));
    }
    c.close();
}

#[test]
fn expert_cache_corpus_matches() {
    let mut r = Replay::new();
    scenario_profile(&mut r);
    scenario_rank(&mut r);
    scenario_open(&mut r);
    scenario_segmented(&mut r);
    scenario_admission(&mut r);
    scenario_copy(&mut r);
    scenario_close(&mut r);

    let golden_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/expert_cache.txt");
    let golden = std::fs::read_to_string(golden_path)
        .expect("golden missing; run tools/build_expert_cache_corpus.sh");
    let want: Vec<&str> = golden.lines().collect();
    assert_eq!(r.out.len(), want.len(), "line count");
    for (i, (got, want)) in r.out.iter().zip(want.iter()).enumerate() {
        if got != want {
            panic!("line {}:\n  cpp : {want}\n  rust: {got}", i + 1);
        }
    }
    assert_eq!(NOT_RESIDENT, -1);
}
