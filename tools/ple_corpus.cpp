// tools/ple_corpus.cpp - the golden for ple_reader.cpp's deterministic core.
//
// ple_reader.cpp is the n-gram table reader: 320M rows that never live in RAM, every row off an unbuffered
// 4 KiB read through platform::DirectFile. What is deterministic here is nearly all of it - the
// set-associative row cache, the per-ticket job build (dedup by aligned page, the 2-page extension for a
// straddler, the sort by offset), the out-of-range zero fill, the slot/pump/finish bookkeeping, the
// fault-injection hold, the keep-alive arithmetic and the stats counters. What is not is the wall clock, and
// this harness replaces it: `now_us()` is a pure read of a virtual clock that only `wait()` moves, and
// `wait()` moves it to the scripted completion time of the earliest outstanding read. So a latency, a
// percentile and a `late_injected` count are reproducible here, and they are in the golden.
//
// The harness #includes the real ple_reader.cpp (the RowCache is in an anonymous namespace) and provides the
// platform definitions itself, so nothing links the real direct_file.cpp and no file is opened.
//
// Build (from the strata-rust checkout; $S is the Strata checkout):
//   S=/home/ramishra/work/LLM/Strata
//   R=/home/ramishra/work/LLM/strata-rust
//   g++ -std=c++20 -w -I$S/include -c $R/tools/ple_corpus.cpp -o $R/target/corpus/plec.o
//   g++ -o $R/target/corpus/plecorpus $R/target/corpus/plec.o
//   ./target/corpus/plecorpus > crates/strata-core/tests/golden/ple_reader.txt
//
// tests/ple_reader_corpus.rs replays the same scenarios through the Rust port; the two line streams must be
// identical. The io_thread worker loop is NOT covered - it is the scheduling the fake clock cannot pin down -
// but every function it calls is, driven directly on the Impl.

// The standard headers first, then the access hack: the harness drives Impl's own methods (pump, finish,
// process, release_delayed, drain, keepalive_due, queue_keepalive, record_latency), which the class keeps
// private. Access does not change layout, and this is the only translation unit.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define private public
#include "ngram/ple_reader.cpp"
#undef private

namespace platform = strata::platform;
namespace ng = strata::ngram;

namespace {

// ---------------------------------------------------------------- the fake platform

double g_clock = 0;  // microseconds; only wait() moves it

// scripted completion latency for a read at this offset - the same function on both sides of the golden
double fake_lat(uint64_t offset) {
    return (double) (((offset * 2654435761ull) >> 7) % 1000);
}

struct Fake {
    bool is_open = false;
    uint64_t sz = 0;
    const std::vector<uint8_t>* data = nullptr;  // the master content for the path this handle opened
    struct P {
        uint64_t tag;
        uint32_t bytes;
        bool ok;
        double at;
    };
    std::vector<P> pending;
    std::vector<platform::Completion> wake;
    std::vector<std::string> submits;  // the read plan, as the fake saw it
    bool fail_submit = false;
    bool fail_read = false;
    uint64_t cap_bytes = 0;  // non-zero: no read returns more than this (the short-read branch)
};

std::map<std::string, std::vector<uint8_t>> g_content;
std::map<void*, Fake*> g_bound;  // one state per DirectFile instance, so scenarios do not share a queue
bool g_fail_alloc = false;

}  // namespace

namespace strata::platform {

DirectFile::DirectFile() : impl_(nullptr) { g_bound[(void*) this] = new Fake(); }

DirectFile::~DirectFile() {}

bool DirectFile::open(const std::string& path, std::string& err) {
    auto it = g_content.find(path);
    if (it == g_content.end()) {
        err = "fake: no such file";
        return false;
    }
    Fake* fake = g_bound[(void*) this];
    fake->data = &it->second;
    fake->sz = it->second.size();
    fake->is_open = true;
    *(Fake**) &impl_ = fake;
    return true;
}

void DirectFile::close() {
    if (auto it = g_bound.find((void*) this); it != g_bound.end()) it->second->is_open = false;
}

bool DirectFile::is_open() const {
    auto it = g_bound.find((void*) this);
    return it != g_bound.end() && it->second->is_open;
}

uint64_t DirectFile::size() const {
    auto it = g_bound.find((void*) this);
    return it == g_bound.end() ? 0 : it->second->sz;
}

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    Fake* fake = g_bound[(void*) this];
    fake->submits.push_back(std::to_string(offset) + ":" + std::to_string(length) + ":" + std::to_string(tag));
    if (fake->fail_submit) {
        err = "fake: submit failed";
        return false;
    }
    uint64_t n = std::min<uint64_t>(length, fake->sz > offset ? fake->sz - offset : 0);
    if (fake->cap_bytes && n > fake->cap_bytes) n = fake->cap_bytes;
    memcpy(buffer, fake->data->data() + offset, n);
    fake->pending.push_back({tag, (uint32_t) n, !fake->fail_read, g_clock + fake_lat(offset)});
    return true;
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    Fake* fake = g_bound[(void*) this];
    // a blocking wait runs the clock on to the earliest outstanding read; a poll moves it a fixed step, which
    // is what lets a completion held back by fault injection ever become due
    if (timeout_ms < 0 && !fake->pending.empty()) {
        double at = 1e300;
        for (const auto& p : fake->pending) at = std::min(at, p.at);
        g_clock = std::max(g_clock, at);
    } else {
        g_clock += 100.0;  // a poll costs a poll interval, which is what lets a held completion become due
    }
    int n = 0;
    for (auto it = fake->wake.begin(); it != fake->wake.end() && n < max;) {
        out[n++] = *it;
        it = fake->wake.erase(it);
    }
    for (auto it = fake->pending.begin(); it != fake->pending.end() && n < max;) {
        if (it->at > g_clock) {
            ++it;
            continue;
        }
        out[n++] = Completion{it->tag, it->bytes, it->ok};
        it = fake->pending.erase(it);
    }
    return n;
}

void DirectFile::wake() {
    g_bound[(void*) this]->wake.push_back(Completion{DirectFile::WAKE_TAG, 0, true});
}

void* DirectFile::alloc_aligned(size_t bytes) {
    if (g_fail_alloc) return nullptr;
    return aligned_alloc(4096, (bytes + 4095) & ~(size_t) 4095);  // the fake never checks alignment
}

void DirectFile::free_aligned(void* p) { free(p); }

double now_us() { return g_clock; }

}  // namespace strata::platform

namespace {

// ---------------------------------------------------------------- the fixture

uint64_t fnv(const std::vector<uint8_t>& bytes) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : bytes) {
        h ^= b;
        h *= 1099511628211ull;
    }
    return h;
}

// the same ground truth ple_reader_test.cpp uses: every row encodes its own index
void expected_row(uint32_t row, uint32_t rb, uint8_t* out) {
    for (uint32_t b = 0; b < rb; ++b) out[b] = (uint8_t) ((row * 2654435761u + b * 97u) >> 7);
    memcpy(out, &row, 4);
}

constexpr uint64_t HEADER = 192;  // the real shard's data offset, so rows are misaligned the same way

std::vector<uint8_t>* make_table(const std::string& key, uint64_t rows, uint32_t rb) {
    std::vector<uint8_t>& out = g_content[key];
    out.assign(HEADER, 0xAB);
    std::vector<uint8_t> r(rb);
    for (uint32_t i = 0; i < rows; ++i) {
        expected_row(i, rb, r.data());
        out.insert(out.end(), r.begin(), r.end());
    }
    return &out;
}

uint64_t g_lcg = 12345;
uint32_t rnd(uint32_t n) {
    g_lcg = g_lcg * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t) ((g_lcg >> 33) % n);
}

void reset_lcg() { g_lcg = 12345; }

// ---------------------------------------------------------------- scenarios

Fake* fake_of(ng::PleReader& rd) { return g_bound[(void*) &rd.impl_->file]; }

// the read plan, as the fake saw it. Long plans are hashed, not printed - the golden is a diff surface, not a
// log; the first 8 entries are printed so a reordering is still visible by eye.
std::string plan(const Fake* f, size_t from) {
    std::vector<uint8_t> all;
    std::string head;
    for (size_t i = from; i < f->submits.size(); ++i) {
        for (char c : f->submits[i]) all.push_back((uint8_t) c);
        all.push_back('|');
        if (i < from + 8) {
            if (i > from) head += ",";
            head += f->submits[i];
        }
    }
    if (f->submits.size() - from <= 8) return head;
    char buf[128];
    std::snprintf(buf, sizeof buf, ",...n=%zu,h=%016llx", f->submits.size() - from, (unsigned long long) fnv(all));
    return head + buf;
}

void stats_line(const char* label, const ng::ReaderStats& st, const std::vector<uint8_t>& out) {
    std::printf("CORPUS|STATS|%s|req=%llu|hits=%llu|dedup=%llu|reads=%llu|bytes=%llu|late=%llu|ka=%llu|"
                "ka_max=%.0f|wait_us=%.0f|submit_us=%.0f|sum_us=%.0f|ring=%zu|p50=%.0f|p99=%.0f|out=%016llx\n",
                label, (unsigned long long) st.requests, (unsigned long long) st.cache_hits,
                (unsigned long long) st.dedup_rows, (unsigned long long) st.reads, (unsigned long long) st.bytes,
                (unsigned long long) st.late_injected, (unsigned long long) st.keepalive_reads, st.keepalive_us_max,
                st.wait_us, st.submit_us, st.read_us_sum, st.read_us.size(), st.percentile(0.5),
                st.percentile(0.99), (unsigned long long) fnv(out));
}

// one ticket through the public API; `out` is the caller's buffer, exactly as the header documents it
bool ticket(ng::PleReader& rd, Fake* f, const std::vector<uint32_t>& rows, std::vector<uint8_t>& out,
            const char* label) {
    out.assign(rows.size() * rd.row_bytes(), 0xCC);
    const size_t from = f->submits.size();
    const auto t = rd.issue(rows.data(), rows.size(), out.data());
    std::string err;
    const bool ok = rd.collect(t, err);
    std::printf("CORPUS|TICKET|%s|plan=%s|ok=%d|err=%s|out=%016llx\n", label, plan(f, from).c_str(), ok ? 1 : 0,
                err.c_str(), (unsigned long long) fnv(out));
    return ok;
}

void open_case(ng::PleReader& rd, std::string& err, const char* label, const char* path, uint64_t off,
               uint64_t rows, uint32_t inflight, uint64_t cache, bool thread = true, uint32_t rb = 90,
               bool fail_alloc = false) {
    err.clear();
    g_fail_alloc = fail_alloc;
    const int ok = rd.open(path, off, rows, inflight, cache, err, thread, rb) ? 1 : 0;
    g_fail_alloc = false;
    std::printf("CORPUS|OPEN|%s|%d|%s|rb=%u|cap=%llu\n", label, ok, err.c_str(), rd.row_bytes(),
                (unsigned long long) rd.cache_capacity());
    rd.close();
}

void scenario_opens() {
    ng::PleReader rd;
    std::string err;
    open_case(rd, err, "inflight0", "t90", 0, 500000, 0, 0);
    open_case(rd, err, "inflight1025", "t90", 0, 500000, 1025, 0);
    open_case(rd, err, "rb0", "t90", 0, 500000, 8, 0, true, 0);
    open_case(rd, err, "rb4097", "t90", 0, 500000, 8, 0, true, 4097);
    open_case(rd, err, "pasteof", "t90", 0, 900000000, 8, 0);
    open_case(rd, err, "missing", "nope", 0, 10, 8, 0);
    open_case(rd, err, "allocfail", "t90", 0, 500000, 8, 0, true, 90, true);
    open_case(rd, err, "ok", "t90", HEADER, 500000, 8, 4096);
}

void scenario_cache() {
    ng::PleReader rd;
    std::string err;
    if (!rd.open("t90", HEADER, 500000, 4, 64, err, false)) {
        std::printf("CORPUS|CACHE|open failed\n");
        return;
    }
    auto& c = rd.impl_->cache;
    std::printf("CORPUS|CACHE|init|sets=%llu|ways=%u|rb=%u|cap=%llu|used=%llu\n", (unsigned long long) c.sets,
                strata::ngram::WAYS, c.rb, (unsigned long long) (c.sets * strata::ngram::WAYS), (unsigned long long) c.used);
    std::string ops;
    for (uint32_t r : {3u, 17u, 1000u, 1001u, 4096u, 8192u, 3u, 99u, 499999u}) {
        const bool hit = c.find(r) != nullptr;
        if (!hit) c.insert(r, rd.impl_->slot_buf(0));
        ops += std::to_string(r) + (hit ? ":hit" : ":miss") + ":" + std::to_string(c.mix(r) % c.sets) + ";";
    }
    std::printf("CORPUS|CACHE|ops|%s|used=%llu|cap=%llu\n", ops.c_str(), (unsigned long long) c.used,
                (unsigned long long) rd.cache_capacity());
    rd.close();

    // overfill: 16 rows of capacity, 4000 inserted - who is left is the round-robin's survivors
    ng::PleReader rd2;
    if (!rd2.open("t90", HEADER, 500000, 4, 16, err, false)) {
        std::printf("CORPUS|CACHE|open2 failed\n");
        return;
    }
    auto& c2 = rd2.impl_->cache;
    std::vector<uint8_t> tmp(90, 7);
    std::string keys;
    for (uint32_t r = 0; r < 4000; ++r) c2.insert(r, tmp.data());
    for (uint32_t r = 0; r < 4000; ++r)
        if (c2.find(r) != nullptr) keys += std::to_string(r) + " ";
    std::printf("CORPUS|CACHE|kept|%s|used=%llu|cap=%llu\n", keys.c_str(), (unsigned long long) c2.used,
                (unsigned long long) rd2.cache_capacity());
    rd2.close();
}

void scenario_pump() {
    // free_slots is a stack seeded high-to-low, so the first slot handed out is the last one
    ng::PleReader rd;
    std::string err;
    if (!rd.open("t90", HEADER, 500000, 4, 0, err, false)) {
        std::printf("CORPUS|PUMP|open failed\n");
        return;
    }
    auto& m = *rd.impl_;
    std::string slots;
    for (const auto s : m.free_slots) { if (!slots.empty()) slots += " "; slots += std::to_string(s); }
    std::printf("CORPUS|PUMP|free|%s\n", slots.c_str());
    std::vector<uint8_t> out;
    ticket(rd, fake_of(rd), {9, 10, 11}, out, "pump");
    std::printf("CORPUS|PUMP|after|free=%zu|queue=%zu|inflight=%zu\n", m.free_slots.size(), m.queue.size(),
                m.inflight.size());
    rd.close();
    std::printf("CORPUS|PUMP|closed|free=%zu|inflight=%zu|cap=%llu\n", m.free_slots.size(), m.inflight.size(),
                (unsigned long long) rd.cache_capacity());
}

void scenario_tickets(const std::string& key, uint64_t rows, uint32_t rb, uint64_t cache_rows, uint32_t inflight,
                      bool delay) {
    ng::PleReader rd;
    std::string err;
    if (!rd.open(key, HEADER, rows, inflight, cache_rows, err, false, rb)) {
        std::printf("CORPUS|TICKET|open failed: %s\n", err.c_str());
        return;
    }
    if (delay) rd.set_injected_delay_us(300);
    Fake* f = fake_of(rd);
    std::vector<uint8_t> out;
    reset_lcg();
    for (int t = 0; t < 8; ++t) {
        std::vector<uint32_t> r(16);
        for (auto& x : r) x = rnd((uint32_t) rows);
        ticket(rd, f, r, out, "decode");
    }
    std::vector<uint32_t> straddle;
    for (uint32_t r = 0; r < rows && straddle.size() < 12; ++r) {
        const uint64_t a = HEADER + (uint64_t) r * rb;
        if (a / 4096 != (a + rb - 1) / 4096) straddle.push_back(r);
    }
    ticket(rd, f, straddle, out, "straddle");
    ticket(rd, f, {5, 5, 6, 7, 5, 0, (uint32_t) rows - 1, (uint32_t) rows, 0xFFFFFFFFu, 44, 45}, out, "dedup");
    std::vector<uint32_t> bulk(2000);
    for (auto& r : bulk) r = rnd((uint32_t) rows);
    ticket(rd, f, bulk, out, "bulk");
    std::vector<uint32_t> a(16), b(16);
    for (auto& r : a) r = rnd((uint32_t) rows);
    for (auto& r : b) r = rnd((uint32_t) rows);
    {
        std::vector<uint8_t> oa(a.size() * rb, 0xCC), ob(b.size() * rb, 0xCC);
        const size_t from = f->submits.size();
        const auto ta = rd.issue(a.data(), a.size(), oa.data());
        const auto tb = rd.issue(b.data(), b.size(), ob.data());
        const bool okb = rd.collect(tb, err);
        const bool oka = rd.collect(ta, err);
        std::printf("CORPUS|TICKET|two-reverse|plan=%s|ok=%d%d|err=%s|out=%016llx,%016llx\n",
                    plan(f, from).c_str(), oka, okb, err.c_str(), (unsigned long long) fnv(oa),
                    (unsigned long long) fnv(ob));
    }
    stats_line(key.c_str(), rd.snapshot(), out);
    rd.close();
}

void scenario_errors() {
    std::vector<uint8_t> out;
    // a read that cannot be queued: the ticket's pending count must still settle, and the queue is cancelled
    ng::PleReader rd;
    std::string err;
    if (!rd.open("t90", HEADER, 500000, 4, 0, err, false)) {
        std::printf("CORPUS|ERR|open failed\n");
        return;
    }
    Fake* f = fake_of(rd);
    f->fail_submit = true;
    ticket(rd, f, {1, 2, 3}, out, "submitfail");
    f->fail_submit = false;
    rd.close();

    ng::PleReader rd2;
    if (!rd2.open("t90", HEADER, 500000, 4, 0, err, false)) {
        std::printf("CORPUS|ERR|open2 failed\n");
        return;
    }
    fake_of(rd2)->fail_read = true;
    ticket(rd2, fake_of(rd2), {1, 2, 3}, out, "readfail");
    fake_of(rd2)->fail_read = false;
    rd2.close();

    // a read that comes back shorter than the window it asked for: a use inside it is a refusal
    ng::PleReader rd3;
    if (!rd3.open("t90", HEADER, 500000, 4, 0, err, false)) {
        std::printf("CORPUS|ERR|open3 failed\n");
        return;
    }
    fake_of(rd3)->cap_bytes = 4096;
    ticket(rd3, fake_of(rd3), {1, 2, 3}, out, "shortread");
    fake_of(rd3)->cap_bytes = 0;
    rd3.close();

    // an unknown ticket, and a completion for a slot that was never handed out
    ng::PleReader rd4;
    if (!rd4.open("t90", HEADER, 500000, 4, 0, err, false)) {
        std::printf("CORPUS|ERR|open4 failed\n");
        return;
    }
    std::string e2;
    const int unk2 = rd4.collect(ng::PleReader::Ticket{7}, e2) ? 1 : 0;
    std::printf("CORPUS|ERR|unknown-ticket|%d|%s\n", unk2, e2.c_str());
    // a completion for a slot that was never handed out: `process` is the only way one arrives, and it is the
    // only place the tag is range-checked - `release_delayed` indexes unguarded because nothing that reaches
    // the delayed list has not already passed that check.
    platform::Completion bad[2] = {platform::Completion{99, 4096, true}, platform::Completion{2, 4096, true}};
    const int bt2 = rd4.impl_->process(bad, 2) ? 1 : 0;
    std::printf("CORPUS|ERR|bad-tag|%d|%s\n", bt2, rd4.impl_->error.c_str());
    rd4.close();
    rd.close();
    rd2.close();
    rd3.close();

    // a wake packet in a batch is skipped, not finished
    ng::PleReader rd5;
    if (!rd5.open("t90", HEADER, 500000, 4, 0, err, false)) {
        std::printf("CORPUS|ERR|open5 failed\n");
        return;
    }
    rd5.impl_->file.wake();
    platform::Completion got[8];
    const int n = rd5.impl_->file.wait(got, 8, 0);
    std::printf("CORPUS|ERR|wake-skip|n=%d|ok=%d\n", n, rd5.impl_->process(got, n) ? 1 : 0);
    rd5.close();
}

void scenario_keepalive() {
    ng::PleReader rd;
    std::string err;
    if (!rd.open("t90", HEADER, 500000, 4, 0, err, false)) {
        std::printf("CORPUS|KA|open failed\n");
        return;
    }
    auto& m = *rd.impl_;
    std::printf("CORPUS|KA|off|due=%.0f\n", m.keepalive_due(g_clock));
    rd.set_keepalive(20.0, 0.5);
    std::printf("CORPUS|KA|caller-thread|due=%.0f\n", m.keepalive_due(g_clock));
    rd.close();

    ng::PleReader rd2;
    if (!rd2.open("t90", HEADER, 500000, 4, 0, err, false)) {
        std::printf("CORPUS|KA|open2 failed\n");
        return;
    }
    rd2.impl_->threaded = true;  // the flag gates the keep-alive; no worker is started, the harness drives it
    rd2.set_keepalive(20.0, 0.5);
    auto& m2 = *rd2.impl_;
    std::string due;
    for (int i = 0; i < 5; ++i) {
        m2.queue_keepalive();
        due += std::to_string(m2.queue.back().offset) + ":" + std::to_string(m2.queue.back().length) + ";";
    }
    std::printf("CORPUS|KA|plan|%s\n", due.c_str());
    std::printf("CORPUS|KA|due|now=%.0f|due=%.0f|window=%.0f\n", g_clock, m2.keepalive_due(g_clock),
                m2.keep_window_us);
    m2.last_issue_us = g_clock;
    m2.last_read_us = g_clock;
    std::printf("CORPUS|KA|due-armed|due=%.0f\n", m2.keepalive_due(g_clock));
    std::printf("CORPUS|KA|due-window-passed|due=%.0f\n", m2.keepalive_due(g_clock + m2.keep_window_us + 1));
    m2.error = "fake";
    std::printf("CORPUS|KA|due-error|due=%.0f\n", m2.keepalive_due(g_clock));
    rd2.close();
}

void scenario_latency_ring() {
    ng::PleReader rd;
    std::string err;
    if (!rd.open("t90", HEADER, 500000, 2, 0, err)) {
        std::printf("CORPUS|RING|open failed\n");
        return;
    }
    for (int i = 0; i < 5; ++i) rd.impl_->record_latency((double) i * 100);
    std::printf("CORPUS|RING|sum=%.0f|n=%zu|p0=%.0f|p50=%.0f|p100=%.0f\n", rd.impl_->stats.read_us_sum,
                rd.impl_->stats.read_us.size(), rd.impl_->stats.percentile(0.0), rd.impl_->stats.percentile(0.5),
                rd.impl_->stats.percentile(1.0));
    rd.impl_->ring_pos = 1;
    rd.impl_->record_latency(55);
    std::printf("CORPUS|RING|overwrite|sum=%.0f|n=%zu|p50=%.0f\n", rd.impl_->stats.read_us_sum,
                rd.impl_->stats.read_us.size(), rd.impl_->stats.percentile(0.5));
    rd.impl_->stats.read_us.clear();
    std::printf("CORPUS|RING|empty|p50=%.0f\n", rd.impl_->stats.percentile(0.5));
    rd.impl_->record_latency(7);
    rd.reset_stats();
    std::printf("CORPUS|RING|reset|sum=%.0f|n=%zu|ringpos=%zu\n", rd.impl_->stats.read_us_sum,
                rd.impl_->stats.read_us.size(), rd.impl_->ring_pos);
    rd.close();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    auto run = [&](const std::string& name) { return only.empty() || only == name; };
    make_table("t90", 500000, 90);
    make_table("t110", 200000, 110);

    if (run("opens")) scenario_opens();
    if (run("cache")) scenario_cache();
    if (run("pump")) scenario_pump();
    if (run("c1")) scenario_tickets("t90", 500000, 90, 0, 1, false);
    if (run("c2")) scenario_tickets("t90", 500000, 90, 4096, 8, false);
    if (run("c3")) scenario_tickets("t110", 200000, 110, 0, 64, false);
    if (run("c4")) scenario_tickets("t90", 500000, 90, 0, 16, true);
    if (run("errors")) scenario_errors();
    if (run("keepalive")) scenario_keepalive();
    if (run("ring")) scenario_latency_ring();
    return 0;
}
