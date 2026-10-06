//! Byte-parity replay of `ple_reader.cpp`'s deterministic core: the row cache, the
//! per-ticket job build, the slot bookkeeping, the fault-injection hold, the
//! keep-alive arithmetic, the latency ring and the stats counters.
//!
//! `tools/ple_corpus.cpp` compiles the real `ple_reader.cpp` and provides the
//! platform layer itself, so no file is opened and no thread runs. The fake's clock
//! is the load-bearing part: `now_us()` is a pure read, and only `wait()` moves it -
//! a blocking wait runs it on to the earliest outstanding read's scripted completion
//! time, a poll moves it a fixed 100 us. That is what makes a latency, a percentile
//! and a `late_injected` count reproducible, so they are in the golden here and not
//! excluded from it.
//!
//! The io_thread worker loop is not covered - it is the scheduling the fake clock
//! cannot pin down - but every function it calls is, driven directly.
//!
//! Regenerate (needs a Strata checkout; build line in the harness header):
//!   ./target/corpus/plecorpus > crates/strata-core/tests/golden/ple_reader.txt

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use strata_core::ngram::{Completion, Io, PleReader};

const GOLDEN: &str = include_str!("golden/ple_reader.txt");
const HEADER: u64 = 192; // the real shard's data offset, so rows are misaligned the same way

// ---------------------------------------------------------------- the fake platform

#[derive(Clone, Copy)]
struct Pending {
    tag: u64,
    bytes: u32,
    ok: bool,
    at: f64,
}

#[derive(Default)]
struct Shared {
    contents: BTreeMap<String, Vec<u8>>,
    path: String, // the table this handle opened
    slab_len: u64,
    clock: f64, // microseconds; only wait() moves it
    submits: Vec<String>,
    pending: Vec<Pending>,
    wakeq: Vec<Completion>,
    fail_submit: bool,
    fail_read: bool,
    fail_alloc: bool,
    cap_bytes: u64,
    is_open: bool,
    slab: Vec<u8>,
}

/// The scripted completion latency for a read at this offset - the same function on
/// both sides of the golden.
fn fake_lat(offset: u64) -> f64 {
    let v = ((offset.wrapping_mul(2_654_435_761)) >> 7) % 1000;
    v as f64
}

#[derive(Default)]
struct Fake {
    st: Rc<RefCell<Shared>>,
}

impl Fake {
    fn new(st: Rc<RefCell<Shared>>) -> Fake {
        Fake { st }
    }
}

impl Io for Fake {
    fn open(&mut self, path: &str) -> Result<(), String> {
        let st = &mut *self.st.borrow_mut();
        let Some(len) = st.contents.get(path).map(Vec::len) else {
            return Err("fake: no such file".to_string());
        };
        st.path = path.to_string();
        st.slab_len = len as u64;
        st.is_open = true;
        Ok(())
    }
    fn close(&mut self) {
        self.st.borrow_mut().is_open = false;
    }
    fn is_open(&self) -> bool {
        self.st.borrow().is_open
    }
    fn size(&self) -> u64 {
        self.st.borrow().slab_len
    }
    fn alloc_slab(&mut self, bytes: usize) -> bool {
        let st = &mut *self.st.borrow_mut();
        if st.fail_alloc {
            return false;
        }
        st.slab = vec![0; bytes];
        true
    }
    fn free_slab(&mut self) {
        self.st.borrow_mut().slab.clear();
    }
    fn slot_read(&mut self, slot: usize, from: usize, out: &mut [u8]) {
        let st = self.st.borrow();
        let off = slot * 2 * 4096 + from;
        out.copy_from_slice(&st.slab[off..off + out.len()]);
    }
    fn submit(&mut self, offset: u64, slot: usize, length: u32, tag: u64) -> Result<(), String> {
        let st = &mut *self.st.borrow_mut();
        st.submits.push(format!("{offset}:{length}:{tag}"));
        if st.fail_submit {
            return Err("fake: submit failed".to_string());
        }
        let path = st.path.clone();
        let n = std::cmp::min(u64::from(length), st.slab_len.saturating_sub(offset));
        let n = if st.cap_bytes > 0 && n > st.cap_bytes {
            st.cap_bytes
        } else {
            n
        };
        let bytes = st
            .contents
            .get(&path)
            .expect("open bound the fake to a table")
            [offset as usize..offset as usize + n as usize]
            .to_vec();
        let off = slot * 2 * 4096;
        st.slab[off..off + n as usize].copy_from_slice(&bytes);
        let at = st.clock + fake_lat(offset);
        st.pending.push(Pending {
            tag,
            bytes: n as u32,
            ok: !st.fail_read,
            at,
        });
        Ok(())
    }
    fn wait(&mut self, out: &mut Vec<Completion>, max: usize, timeout_ms: i32) {
        let st = &mut *self.st.borrow_mut();
        // a blocking wait runs the clock on to the earliest outstanding read; a poll
        // moves it a fixed step, which is what lets a held completion ever become due
        if timeout_ms < 0 && !st.pending.is_empty() {
            let at = st
                .pending
                .iter()
                .map(|p| p.at)
                .fold(f64::INFINITY, f64::min);
            st.clock = f64::max(st.clock, at);
        } else {
            st.clock += 100.0;
        }
        while !st.wakeq.is_empty() && out.len() < max {
            out.push(st.wakeq.remove(0));
        }
        let mut keep = Vec::new();
        for p in st.pending.drain(..) {
            if out.len() < max && p.at <= st.clock {
                out.push(Completion {
                    tag: p.tag,
                    bytes: p.bytes,
                    ok: p.ok,
                });
            } else {
                keep.push(p);
            }
        }
        st.pending = keep;
    }
    fn wake(&mut self) {
        self.st.borrow_mut().wakeq.push(Completion {
            tag: u64::MAX,
            bytes: 0,
            ok: true,
        });
    }
    fn now_us(&self) -> f64 {
        self.st.borrow().clock
    }
}

// ---------------------------------------------------------------- the fixture

fn fnv(bytes: &[u8]) -> u64 {
    strata_core::fnv(bytes)
}

// the same ground truth ple_reader_test.cpp uses: every row encodes its own index
fn expected_row(row: u32, rb: u32, out: &mut [u8]) {
    for b in 0..rb {
        out[b as usize] =
            (((row.wrapping_mul(2_654_435_761)).wrapping_add(b.wrapping_mul(97))) >> 7) as u8;
    }
    out[..4].copy_from_slice(&row.to_le_bytes());
}

fn make_table(shared: &Rc<RefCell<Shared>>, key: &str, rows: u64, rb: u32) {
    let mut data = vec![0xAB; HEADER as usize];
    let mut r = vec![0u8; rb as usize];
    for i in 0..rows as u32 {
        expected_row(i, rb, &mut r);
        data.extend_from_slice(&r);
    }
    shared.borrow_mut().contents.insert(key.to_string(), data);
}

struct Lcg(u64);
impl Lcg {
    fn reset(&mut self) {
        self.0 = 12345;
    }
    fn next(&mut self, n: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % u64::from(n)) as u32
    }
}

// ---------------------------------------------------------------- scenarios

/// Everything the harness prints, in the order it prints it.
struct Replay {
    shared: Rc<RefCell<Shared>>,
    out: Vec<String>,
    lcg: Lcg,
}

impl Replay {
    fn plan(&self, from: usize) -> String {
        let st = self.shared.borrow();
        let all: Vec<u8> = st.submits[from..]
            .iter()
            .flat_map(|s| s.bytes().chain(*b"|"))
            .collect();
        let head: Vec<String> = st.submits[from..].iter().take(8).cloned().collect();
        if st.submits.len() - from <= 8 {
            return head.join(",");
        }
        format!(
            "{},...n={},h={:016x}",
            head.join(","),
            st.submits.len() - from,
            fnv(&all)
        )
    }

    fn stats_line(&mut self, label: &str, rd: &PleReader<Fake>, out: u64) {
        let st = rd.snapshot();
        self.out.push(format!(
            "CORPUS|STATS|{label}|req={}|hits={}|dedup={}|reads={}|bytes={}|late={}|ka={}|ka_max={:.0}|\
             wait_us={:.0}|submit_us={:.0}|sum_us={:.0}|ring={}|p50={:.0}|p99={:.0}|out={:016x}",
            st.requests,
            st.cache_hits,
            st.dedup_rows,
            st.reads,
            st.bytes,
            st.late_injected,
            st.keepalive_reads,
            st.keepalive_us_max,
            st.wait_us,
            st.submit_us,
            st.read_us_sum,
            st.read_us.len(),
            st.percentile(0.5),
            st.percentile(0.99),
            out
        ));
    }

    fn ticket(&mut self, rd: &mut PleReader<Fake>, rows: &[u32], label: &str) -> u64 {
        let from = self.shared.borrow().submits.len();
        let out = rd.register_output(rows.len());
        rd.output_mut(out)
            .unwrap()
            .iter_mut()
            .for_each(|b| *b = 0xCC);
        let t = rd.issue(rows, out);
        let ok = rd.collect(t);
        let hash = fnv(rd.output(out).unwrap());
        let okflag = ok.is_ok();
        let err = ok.err().unwrap_or_default();
        self.out.push(format!(
            "CORPUS|TICKET|{label}|plan={}|ok={}|err={}|out={hash:016x}",
            self.plan(from),
            okflag as u8,
            err
        ));
        hash
    }

    fn open_case(&mut self, c: &Case) {
        let mut rd = PleReader::new(Fake::new(Rc::clone(&self.shared)));
        self.shared.borrow_mut().fail_alloc = c.fail_alloc;
        let ok = rd.open(c.path, c.off, c.rows, c.inflight, c.cache, c.rb);
        self.shared.borrow_mut().fail_alloc = false;
        self.out.push(format!(
            "CORPUS|OPEN|{}|{}|{}|rb={}|cap={}",
            c.label,
            i64::from(ok.is_ok()),
            ok.err().unwrap_or_default(),
            rd.row_bytes(),
            rd.cache_capacity()
        ));
        rd.close();
    }
}

/// One `open` case: the arguments, and the label the golden prints them under.
#[derive(Clone, Default)]
struct Case {
    label: &'static str,
    path: &'static str,
    off: u64,
    rows: u64,
    inflight: u32,
    cache: u64,
    rb: u32,
    fail_alloc: bool,
}

fn scenario_opens(r: &mut Replay) {
    let big = Case {
        path: "t90",
        rows: 500_000,
        inflight: 8,
        rb: 90, // the C++ harness's default row size
        ..Case::default()
    };
    r.open_case(&Case {
        label: "inflight0",
        inflight: 0,
        ..big.clone()
    });
    r.open_case(&Case {
        label: "inflight1025",
        inflight: 1025,
        ..big.clone()
    });
    r.open_case(&Case {
        label: "rb0",
        rb: 0,
        ..big.clone()
    });
    r.open_case(&Case {
        label: "rb4097",
        rb: 4097,
        ..big.clone()
    });
    r.open_case(&Case {
        label: "pasteof",
        rows: 900_000_000,
        ..big.clone()
    });
    r.open_case(&Case {
        label: "missing",
        path: "nope",
        rows: 10,
        ..big.clone()
    });
    r.open_case(&Case {
        label: "allocfail",
        fail_alloc: true,
        ..big.clone()
    });
    r.open_case(&Case {
        label: "ok",
        off: HEADER,
        cache: 4096,
        ..big
    });
}

fn scenario_cache(r: &mut Replay) {
    let mut rd = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd.open("t90", HEADER, 500_000, 4, 64, 90).unwrap();
    r.out.push(format!(
        "CORPUS|CACHE|init|sets={}|ways=8|rb={}|cap={}|used={}",
        rd.cache_sets(),
        rd.cache_rb(),
        rd.cache_capacity(),
        rd.cache_size()
    ));
    let mut ops = String::new();
    let zeros = vec![0u8; 90];
    for row in [3u32, 17, 1000, 1001, 4096, 8192, 3, 99, 499999] {
        let (hit, set) = rd.cache_probe(row);
        if !hit {
            rd.cache_insert(row, &zeros);
        }
        ops.push_str(&format!(
            "{}:{}:{};",
            row,
            if hit { "hit" } else { "miss" },
            set
        ));
    }
    r.out.push(format!(
        "CORPUS|CACHE|ops|{ops}|used={}|cap={}",
        rd.cache_size(),
        rd.cache_capacity()
    ));
    rd.close();

    // overfill: 16 rows of capacity, 4000 inserted - who is left is the round-robin's survivors
    let mut rd2 = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd2.open("t90", HEADER, 500_000, 4, 16, 90).unwrap();
    for row in 0..4000u32 {
        rd2.cache_insert(row, &zeros);
    }
    let mut kept = String::new();
    for row in 0..4000u32 {
        if rd2.cache_probe(row).0 {
            kept.push_str(&format!("{row} "));
        }
    }
    r.out.push(format!(
        "CORPUS|CACHE|kept|{kept}|used={}|cap={}",
        rd2.cache_size(),
        rd2.cache_capacity()
    ));
    rd2.close();
}

fn scenario_pump(r: &mut Replay) {
    let mut rd = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd.open("t90", HEADER, 500_000, 4, 0, 90).unwrap();
    let slots: Vec<String> = rd.free_slots().iter().map(|s| s.to_string()).collect();
    r.out.push(format!("CORPUS|PUMP|free|{}", slots.join(" ")));
    r.ticket(&mut rd, &[9, 10, 11], "pump");
    r.out.push(format!(
        "CORPUS|PUMP|after|free={}|queue={}|inflight={}",
        rd.free_slots().len(),
        rd.queue_len(),
        rd.inflight_len()
    ));
    rd.close();
    r.out.push(format!(
        "CORPUS|PUMP|closed|free={}|inflight={}|cap={}",
        rd.free_slots().len(),
        rd.inflight_len(),
        rd.cache_capacity()
    ));
}

fn scenario_tickets(
    r: &mut Replay,
    key: &str,
    rows: u64,
    rb: u32,
    cache_rows: u64,
    inflight: u32,
    delay: bool,
) {
    let mut rd = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd.open(key, HEADER, rows, inflight, cache_rows, rb)
        .unwrap();
    if delay {
        rd.set_injected_delay_us(300.0);
    }
    let from = r.shared.borrow().submits.len();
    r.lcg.reset();
    r.lcg.reset();
    // the harness prints each ticket as it goes, so the plan window has to be taken per ticket
    let tickets: Vec<Vec<u32>> = (0..8)
        .map(|_| (0..16).map(|_| r.lcg.next(rows as u32)).collect())
        .collect();
    for t in &tickets {
        let f = r.shared.borrow().submits.len();
        let out = rd.register_output(t.len());
        rd.output_mut(out)
            .unwrap()
            .iter_mut()
            .for_each(|b| *b = 0xCC);
        let tk = rd.issue(t, out);
        let ok = rd.collect(tk);
        r.out.push(format!(
            "CORPUS|TICKET|decode|plan={}|ok={}|err={}|out={:016x}",
            r.plan(f),
            ok.is_ok() as u8,
            ok.err().unwrap_or_default(),
            fnv(rd.output(out).unwrap())
        ));
    }
    let mut straddle: Vec<u32> = Vec::new();
    for row in 0..rows as u32 {
        let a = HEADER + u64::from(row) * u64::from(rb);
        if a / 4096 != (a + u64::from(rb) - 1) / 4096 {
            straddle.push(row);
            if straddle.len() >= 12 {
                break;
            }
        }
    }
    r.ticket(&mut rd, &straddle, "straddle");
    let _ = r.ticket(
        &mut rd,
        &[
            5,
            5,
            6,
            7,
            5,
            0,
            rows as u32 - 1,
            rows as u32,
            u32::MAX,
            44,
            45,
        ],
        "dedup",
    );
    let bulk: Vec<u32> = (0..2000).map(|_| r.lcg.next(rows as u32)).collect();
    let last = r.ticket(&mut rd, &bulk, "bulk");
    let a: Vec<u32> = (0..16).map(|_| r.lcg.next(rows as u32)).collect();
    let b: Vec<u32> = (0..16).map(|_| r.lcg.next(rows as u32)).collect();
    {
        let f = r.shared.borrow().submits.len();
        let oa = rd.register_output(a.len());
        let ob = rd.register_output(b.len());
        rd.output_mut(oa)
            .unwrap()
            .iter_mut()
            .for_each(|x| *x = 0xCC);
        rd.output_mut(ob)
            .unwrap()
            .iter_mut()
            .for_each(|x| *x = 0xCC);
        let ta = rd.issue(&a, oa);
        let tb = rd.issue(&b, ob);
        let okb = rd.collect(tb);
        let oka = rd.collect(ta);
        r.out.push(format!(
            "CORPUS|TICKET|two-reverse|plan={}|ok={}{}|err={}|out={:016x},{:016x}",
            r.plan(f),
            oka.is_ok() as u8,
            okb.is_ok() as u8,
            okb.err().or(oka.err()).unwrap_or_default(),
            fnv(rd.output(oa).unwrap()),
            fnv(rd.output(ob).unwrap())
        ));
    }
    r.stats_line(key, &rd, last);
    let _ = from;
    rd.close();
}

fn scenario_errors(r: &mut Replay) {
    let mut rd = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd.open("t90", HEADER, 500_000, 4, 0, 90).unwrap();
    r.shared.borrow_mut().fail_submit = true;
    r.ticket(&mut rd, &[1, 2, 3], "submitfail");
    r.shared.borrow_mut().fail_submit = false;
    rd.close();

    let mut rd2 = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd2.open("t90", HEADER, 500_000, 4, 0, 90).unwrap();
    r.shared.borrow_mut().fail_read = true;
    r.ticket(&mut rd2, &[1, 2, 3], "readfail");
    r.shared.borrow_mut().fail_read = false;
    rd2.close();

    // a read that comes back shorter than the window it asked for: a use inside it is a refusal
    let mut rd3 = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd3.open("t90", HEADER, 500_000, 4, 0, 90).unwrap();
    r.shared.borrow_mut().cap_bytes = 4096;
    r.ticket(&mut rd3, &[1, 2, 3], "shortread");
    r.shared.borrow_mut().cap_bytes = 0;
    rd3.close();

    // an unknown ticket, and a completion for a slot that was never handed out
    let mut rd4 = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd4.open("t90", HEADER, 500_000, 4, 0, 90).unwrap();
    let unk = rd4.collect(strata_core::ngram::Ticket(7));
    r.out.push(format!(
        "CORPUS|ERR|unknown-ticket|{}|{}",
        i64::from(unk.is_ok()),
        unk.err().unwrap_or_default()
    ));
    let bad = [
        Completion {
            tag: 99,
            bytes: 4096,
            ok: true,
        },
        Completion {
            tag: 2,
            bytes: 4096,
            ok: true,
        },
    ];
    let bt = rd4.process(&bad);
    r.out.push(format!(
        "CORPUS|ERR|bad-tag|{}|{}",
        i64::from(bt),
        rd4.error()
    ));
    rd4.close();
    rd.close();
    rd2.close();
    rd3.close();

    // a wake packet in a batch is skipped, not finished
    let mut rd5 = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd5.open("t90", HEADER, 500_000, 4, 0, 90).unwrap();
    rd5.io_mut().wake();
    let mut got = Vec::new();
    rd5.io_mut().wait(&mut got, 8, 0);
    let ok = rd5.process(&got);
    r.out.push(format!(
        "CORPUS|ERR|wake-skip|n={}|ok={}",
        got.len(),
        ok as u8
    ));
    rd5.close();
}

fn scenario_keepalive(r: &mut Replay) {
    let mut rd = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd.open("t90", HEADER, 500_000, 4, 0, 90).unwrap();
    let now = rd.io().now_us();
    r.out
        .push(format!("CORPUS|KA|off|due={:.0}", rd.keepalive_due(now)));
    rd.set_keepalive(20.0, 0.5);
    let now = rd.io().now_us();
    r.out.push(format!(
        "CORPUS|KA|caller-thread|due={:.0}",
        rd.keepalive_due(now)
    ));
    rd.close();

    let mut rd2 = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd2.open("t90", HEADER, 500_000, 4, 0, 90).unwrap();
    rd2.set_threaded(true); // the flag gates the keep-alive; there is no worker on this arm
    rd2.set_keepalive(20.0, 0.5);
    let mut due = String::new();
    for _ in 0..5 {
        rd2.queue_keepalive();
        let (off, len) = rd2.queue_back();
        due.push_str(&format!("{off}:{len};"));
    }
    r.out.push(format!("CORPUS|KA|plan|{due}"));
    let now = rd2.io().now_us();
    r.out.push(format!(
        "CORPUS|KA|due|now={now:.0}|due={:.0}|window={:.0}",
        rd2.keepalive_due(now),
        rd2.keep_window_us()
    ));
    rd2.arm(now);
    r.out.push(format!(
        "CORPUS|KA|due-armed|due={:.0}",
        rd2.keepalive_due(now)
    ));
    let w = rd2.keep_window_us() + 1.0;
    r.out.push(format!(
        "CORPUS|KA|due-window-passed|due={:.0}",
        rd2.keepalive_due(now + w)
    ));
    rd2.set_error("fake");
    r.out.push(format!(
        "CORPUS|KA|due-error|due={:.0}",
        rd2.keepalive_due(now)
    ));
    rd2.close();
}

fn scenario_latency_ring(r: &mut Replay) {
    let mut rd = PleReader::new(Fake::new(Rc::clone(&r.shared)));
    rd.open("t90", HEADER, 500_000, 2, 0, 90).unwrap();
    for i in 0..5 {
        rd.record_latency(f64::from(i * 100));
    }
    let st = rd.stats();
    r.out.push(format!(
        "CORPUS|RING|sum={:.0}|n={}|p0={:.0}|p50={:.0}|p100={:.0}",
        st.read_us_sum,
        st.read_us.len(),
        st.percentile(0.0),
        st.percentile(0.5),
        st.percentile(1.0)
    ));
    rd.set_ring_pos(1);
    rd.record_latency(55.0);
    let st = rd.stats();
    r.out.push(format!(
        "CORPUS|RING|overwrite|sum={:.0}|n={}|p50={:.0}",
        st.read_us_sum,
        st.read_us.len(),
        st.percentile(0.5)
    ));
    rd.clear_read_us();
    r.out.push(format!(
        "CORPUS|RING|empty|p50={:.0}",
        rd.stats().percentile(0.5)
    ));
    rd.record_latency(7.0);
    rd.reset_stats();
    let st = rd.stats();
    r.out.push(format!(
        "CORPUS|RING|reset|sum={:.0}|n={}|ringpos={}",
        st.read_us_sum,
        st.read_us.len(),
        rd.ring_pos()
    ));
    rd.close();
}

#[test]
fn ple_reader_corpus_matches() {
    let shared = Rc::new(RefCell::new(Shared::default()));
    make_table(&shared, "t90", 500_000, 90);
    make_table(&shared, "t110", 200_000, 110);
    let mut r = Replay {
        shared,
        out: Vec::new(),
        lcg: Lcg(0),
    };

    scenario_opens(&mut r);
    scenario_cache(&mut r);
    scenario_pump(&mut r);
    scenario_tickets(&mut r, "t90", 500_000, 90, 0, 1, false);
    scenario_tickets(&mut r, "t90", 500_000, 90, 4096, 8, false);
    scenario_tickets(&mut r, "t110", 200_000, 110, 0, 64, false);
    scenario_tickets(&mut r, "t90", 500_000, 90, 0, 16, true);
    scenario_errors(&mut r);
    scenario_keepalive(&mut r);
    scenario_latency_ring(&mut r);

    let golden: Vec<&str> = GOLDEN.lines().collect();
    assert_eq!(
        golden.len(),
        r.out.len(),
        "golden has {} lines, the replay made {}",
        golden.len(),
        r.out.len()
    );
    for (i, (g, m)) in golden.iter().zip(r.out.iter()).enumerate() {
        if g != m {
            panic!("line {}:\n  cpp : {g}\n  rust: {m}", i + 1);
        }
    }
}
