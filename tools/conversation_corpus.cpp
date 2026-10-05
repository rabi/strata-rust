// Golden-trace generator for the conversation snapshot contract.
//
// It compiles the REAL `src/core/conversation_state.cpp` and
// `src/core/conversation_snapshot.cpp` from a Strata checkout against stub CUDA
// symbols, runs the same fixtures as `src/core/conversation_validation_test.cpp`
// (plus the transfer paths through `--wrap=cudaMemcpy`), and prints every
// intermediate the Rust port claims to reproduce: byte estimates, error strings,
// rejection verdicts, fnv1a64 fingerprints of every buffer after every phase.
//
// Build (see shim/build.sh's sibling notes in docs/MIGRATION.md):
//   g++ -std=c++20 -w -I$STRATA/include -I$cudastub -DCONVERSATION_TEST_TRANSFERS \
//       -c tools/conversation_corpus.cpp -o corpus.o
//   g++ -o corpus corpus.o st.o sn.o stubs.o \
//       -Wl,--wrap=cudaMemcpy -Wl,--wrap=cudaDeviceSynchronize -Wl,--wrap=cudaGetLastError
//
// Output: one `CORPUS|<kind>|k=v|...` line per observation, byte-identical
// across runs and machines; the Rust test replays it against its own port.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <cuda_runtime.h>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#define CONVERSATION_TEST_TRANSFERS 1

namespace {
// ---------------------------------------------------------------- transfer injection
int copy_calls = 0, sync_calls = 0, fail_copy = 0, fail_sync = 0;
size_t copied_bytes = 0;
// Every buffer the fixtures own, so each injected transfer can be reported as
// (which buffer, at what offset, how many bytes) - the plan the Rust port must
// reproduce. Unregistered destinations are snapshot-side segments; the byte
// count and the sequence position still pin them down.
struct Region { const uint8_t* base; size_t len; const char* name; };
std::vector<Region> regions;
void reg(const void* p, size_t n, const char* name) {
    if (!p || !n) return;
    regions.push_back({static_cast<const uint8_t*>(p), n, name});
}
std::string where(const void* p, size_t n) {
    if (!p) return "null";
    auto* b = static_cast<const uint8_t*>(p);
    for (const auto& r : regions)
        if (b >= r.base && b < r.base + r.len && b + n <= r.base + r.len)
            return std::string(r.name) + ":" + std::to_string(b - r.base);
    return "image";
}
}  // namespace
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    ++copy_calls;
    if (copy_calls == fail_copy) {
        std::printf("CORPUS|COPY|n=%d|bytes=%zu|dst=%s|src=%s|FAILED\n", copy_calls, n,
                    where(dst, n).c_str(), where(src, n).c_str());
        return cudaErrorInvalidValue;
    }
    copied_bytes += n;
    std::printf("CORPUS|COPY|n=%d|bytes=%zu|dst=%s|src=%s\n", copy_calls, n,
                where(dst, n).c_str(), where(src, n).c_str());
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    return ++sync_calls == fail_sync ? cudaErrorUnknown : cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetLastError() { return cudaSuccess; }

namespace {
using namespace strata::core;

std::vector<uint8_t> flat(const ConversationBuffer& b) {
    std::vector<uint8_t> v(b.size(), 0);
    if (!b.read(v.data(), 0, b.size())) std::exit(4);
    return v;
}

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '|') o += "%7C";
        else if (c == '\n') o += "%0A";
        else o += c;
    }
    return o;
}

void line(const std::string& s) { std::printf("CORPUS|%s\n", s.c_str()); }

std::string yesno(bool b) { return b ? "1" : "0"; }
std::string restore_name(ConversationRestore r) {
    return r == ConversationRestore::restored ? "restored"
           : r == ConversationRestore::invalid ? "invalid" : "transfer_failed";
}

// ---------------------------------------------------------------- fixtures (as the C++ test)
struct Pools {
    QsaState st;
    std::array<std::vector<uint8_t>, 8> data;
    Pools(const ModelGeometry& g, int format) {
        st.max_cells = 96; st.n_pages = st.n_slots = 24; st.idx_pooled_rows = 26;
        st.kv_int8 = format == 1; st.kv_q4 = format == 2;
        const size_t per = format == 2 ? strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                                       : g.head_dim * (format == 1 ? 1 : 2);
        data[0].resize(96 * g.n_head_kv * per, 0xa5); data[1] = data[0];
        data[2].resize(format == 1 ? 96 * g.n_head_kv * (g.head_dim / 64) * 2 : 0, 0xa5); data[3] = data[2];
        data[4].resize(26 * g.idx_key_dim * 4, 0xa5);
        data[5].resize(3 * g.idx_key_dim * 4, 0xa5);
        data[6].resize(g.idx_key_dim * 4, 0xa5); data[7].resize(4, 0xa5);
        if (format == 2) { st.k_q4 = data[0].data(); st.v_q4 = data[1].data(); }
        else if (format == 1) {
            st.k_q = (int8_t*) data[0].data(); st.v_q = (int8_t*) data[1].data();
            st.k_scale = (uint16_t*) data[2].data(); st.v_scale = (uint16_t*) data[3].data();
        } else { st.k_pool = (uint16_t*) data[0].data(); st.v_pool = (uint16_t*) data[1].data(); }
        st.idx_pooled = (float*) data[4].data(); st.idx_tail = (float*) data[5].data();
        st.idx_dead = (float*) data[6].data(); st.idx_block_pos = (int32_t*) data[7].data();
    }
    ConversationKv image(const ModelGeometry& g, bool index) const {
        ConversationKv k;
        k.format = qsa_kv_format(st); k.cells = 12; k.page_size = 4;
        k.heads = g.n_head_kv; k.head_dim = g.head_dim; k.idx_dim = g.idx_key_dim;
        k.pooled_rows = index ? 3 : 0;
        k.k.resize(data[0].size() / 8, 13); k.v = k.k;
        k.k_scale.resize(data[2].size() / 8, 13); k.v_scale = k.k_scale;
        k.pooled.resize(k.pooled_rows * g.idx_key_dim * 4, 13);
        return k;
    }
};

// ---------------------------------------------------------------- the state under test
inline ModelGeometry fixture_geometry(int format, int experts, bool zero_qsa, bool ple) {
    (void) format; (void) ple;
    ModelGeometry g;
    g.n_layers = zero_qsa ? 3 : 8; g.n_expert = experts;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    return g;
}

struct World {
    ModelGeometry g;
    int format = 0, experts = 256;
    bool zero_qsa = false, ple = false;
    ConversationStateSizes z;
    Pools first, last, draft;
    std::array<QsaState, 2> layers;
    std::vector<uint8_t> gdn, history;
    SessionState ss;
    SavedConversation image;

    World(int f, int ex, bool zq, bool pl)
        : g(fixture_geometry(f, ex, zq, pl)), format(f), experts(ex), zero_qsa(zq), ple(pl),
          first(g, f), last(g, f), draft(g, f), layers{first.st, last.st} {}

    bool pristine(const std::vector<uint8_t>& b) {
        return std::all_of(b.begin(), b.end(), [](uint8_t v) { return v == 0xa5; });
    }
    // fingerprint of every byte the session owns: the "did an invalid restore touch
    // anything" predicate, as a number instead of a bool
    uint64_t state_fingerprint() {
        uint64_t h = 1469598103934665603ull;
        h = fnv(gdn.data(), gdn.size(), h);
        h = fnv(history.data(), history.size(), h);
        for (auto* p : {&first, &last, &draft})
            for (auto& b : p->data) h = fnv(b.data(), b.size(), h);
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        h = fnv(prev, sizeof(prev), h);
        return h;
    }
    // name every byte-range the session owns, so the injected transfers can be
    // printed as (buffer, offset, count) instead of as a bare byte count
    void register_regions() {
        regions.clear();
        reg(ss.gdn_state, gdn.size(), "gdn");
        reg(ss.ple_hist, history.size(), "ple");
        Pools* pools[3] = {&first, &last, &draft};
        static const char* layer[3] = {"L0", "L1", "DRAFT"};
        static const char* pool[4] = {"k", "v", "k_scale", "v_scale"};
        static const char* extra[4] = {"idx_pooled", "idx_tail", "idx_dead", "idx_block_pos"};
        // names outlive this function: the region registry keeps the pointer
        // deque, not vector: the registry keeps these c_str() pointers and a
        // vector reallocation would invalidate them
        static std::deque<std::string> names;
        names.clear();
        for (int p = 0; p < 3; ++p) {
            for (int i = 0; i < 4; ++i) {
                names.push_back(std::string(layer[p]) + ":" + pool[i]);
                reg(pools[p]->data[i].data(), pools[p]->data[i].size(), names.back().c_str());
            }
            for (int i = 0; i < 4; ++i) {
                names.push_back(std::string(layer[p]) + ":" + extra[i]);
                reg(pools[p]->data[4 + i].data(), pools[p]->data[4 + i].size(), names.back().c_str());
            }
        }
    }

    void reset_state() {
        for (auto* p : {&first, &last, &draft}) for (auto& b : p->data) std::fill(b.begin(), b.end(), 0xa5);
        std::fill(gdn.begin(), gdn.end(), 0xa5);
        std::fill(history.begin(), history.end(), 0xa5);
        ss.ple_prev[0] = ss.ple_prev[1] = -1;
        copy_calls = sync_calls = fail_copy = fail_sync = 0;
        copied_bytes = 0;
    }

    ConversationCheckpoint checkpoint(size_t tokens) {
        ConversationCheckpoint c;
        c.ids.resize(tokens);
        for (size_t i = 0; i < tokens; ++i) c.ids[i] = (int32_t) i + 1;
        c.imgs = {{1, 44}};
        c.gdn.resize(z.gdn, 13); c.ple.resize(ple ? z.ple : 0, 13);
        c.tails.resize(g.n_qsa_layers() * z.tail, 13); c.dead.resize(g.n_qsa_layers() * z.dead, 13);
        c.block_pos.resize(g.n_qsa_layers() * z.block_pos, 13);
        return c;
    }

    void sizes(const ModelGeometry& geo, const char* tag) {
        ConversationStateSizes z2;
        std::string err;
        bool ok = conversation_state_sizes(geo, z2, err);
        line(std::string("SIZES|tag=") + tag + "|ok=" + yesno(ok) + "|err=" + esc(err) +
             "|gdn=" + std::to_string(z2.gdn) + "|ple=" + std::to_string(z2.ple) +
             "|tail=" + std::to_string(z2.tail) + "|dead=" + std::to_string(z2.dead) +
             "|block_pos=" + std::to_string(z2.block_pos));
        ConversationStateSizes z3;
        std::string err2;
        bool ok2 = conversation_session_sizes(geo, ss, z3, err2);
        line(std::string("SESSION_SIZES|tag=") + tag + "|ok=" + yesno(ok2) + "|err=" + esc(err2) +
             "|gdn=" + std::to_string(z3.gdn) + "|ple=" + std::to_string(z3.ple) +
             "|tail=" + std::to_string(z3.tail) + "|dead=" + std::to_string(z3.dead) +
             "|block_pos=" + std::to_string(z3.block_pos));
    }

    void reject(const std::function<void(SavedConversation&)>& mutate, const char* label) {
        auto bad = image;
        mutate(bad);
        std::string err;
        auto r = conversation_snapshot_restore(bad, ss, g, draft.st, err);
        uint64_t fp = state_fingerprint();
        line(std::string("REJECT|label=") + label + "|restore=" + restore_name(r) +
             "|err=" + esc(err) + "|fingerprint=" + std::to_string(fp));
    }

    void run();
};

void World::run() {
    std::string err;
    std::printf("CORPUS|FIX|format=%d|experts=%d|zero_qsa=%d|ple=%d\n", format, experts, zero_qsa ? 1 : 0, ple ? 1 : 0);
    bool ok = conversation_state_sizes(g, z, err);
    line("SIZES|tag=fixture|ok=" + yesno(ok) + "|err=" + esc(err) + "|gdn=" + std::to_string(z.gdn) +
         "|ple=" + std::to_string(z.ple) + "|tail=" + std::to_string(z.tail) + "|dead=" +
         std::to_string(z.dead) + "|block_pos=" + std::to_string(z.block_pos));

    gdn.resize(z.gdn, 0xa5); history.resize(ple ? z.ple : 0, 0xa5);
    ss.max_cells = 96; ss.gdn_state = (float*) gdn.data();
    ss.layer_hi = g.n_layers; ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = g.n_qsa_layers();
    ss.ple_hist = ple ? (float*) history.data() : nullptr;
    ss.qsa_states = zero_qsa ? nullptr : layers.data();
    image.geometry = {g.n_embd, g.n_layers,   g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
                      g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim, g.n_head,
                      g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim, g.hc, g.hc_lr,
                      g.n_expert, g.n_ff};
    image.layer_hi = g.n_layers;
    image.live = checkpoint(9); image.checkpoints.push_back(checkpoint(5));
    image.kv.reserve((size_t) g.n_qsa_layers() + 1);
    if (!zero_qsa) { image.kv.push_back(first.image(g, true)); image.kv.push_back(last.image(g, true)); }
    image.kv.push_back(draft.image(g, false));

    ok = conversation_snapshot_validate(image, ss, g, draft.st, err);
    line("VALIDATE|ok=" + yesno(ok) + "|err=" + esc(err) + "|image_bytes=" + std::to_string(image.bytes()));
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, true};
    size_t estimate = 0;
    ok = conversation_snapshot_bytes(view, ss, g, draft.st, estimate, err);
    line("ESTIMATE|ok=" + yesno(ok) + "|err=" + esc(err) + "|bytes=" + std::to_string(estimate) +
         "|image_bytes=" + std::to_string(image.bytes()));
    size_t capture_estimate = 0;
    ok = conversation_snapshot_capture_bytes({}, view, ss, g, draft.st, capture_estimate, err);
    line("CAPTURE_ESTIMATE|ok=" + yesno(ok) + "|err=" + esc(err) + "|bytes=" + std::to_string(capture_estimate));

    // checkpoint validation, whole and carved
    ok = conversation_checkpoint_validate(image.live, ss, g, err);
    line("CHECKPOINT_VALIDATE|ok=" + yesno(ok) + "|err=" + esc(err));
    if (!zero_qsa) {
        SessionState stage = ss;
        stage.layer_lo = 4; stage.gdn_alloc = 3; stage.qsa_ord0 = 1; stage.qsa_alloc = 1;
        ConversationStateSizes zs;
        ok = conversation_session_sizes(g, stage, zs, err);
        line("CARVE|ok=" + yesno(ok) + "|err=" + esc(err) + "|gdn=" + std::to_string(zs.gdn) +
             "|tail=" + std::to_string(zs.tail) + "|dead=" + std::to_string(zs.dead) +
             "|block_pos=" + std::to_string(zs.block_pos));
        ok = conversation_checkpoint_validate(image.live, stage, g, err);
        line("CARVE_CHECKPOINT|whole_rejected=" + yesno(!ok) + "|err=" + esc(err));
        ConversationCheckpoint part = image.live;
        part.gdn.resize(zs.gdn); part.tails.resize(zs.tail); part.dead.resize(zs.dead);
        part.block_pos.resize(zs.block_pos);
        ok = conversation_checkpoint_validate(part, stage, g, err);
        line("CARVE_CHECKPOINT|part_ok=" + yesno(ok) + "|err=" + esc(err));
        ok = conversation_snapshot_validate(image, stage, g, draft.st, err);
        line("CARVE_SNAPSHOT|rejected=" + yesno(!ok) + "|err=" + esc(err));
        stage.qsa_alloc = 2;
        ConversationStateSizes junk;
        ok = conversation_session_sizes(g, stage, junk, err);
        line("CARVE|qsa_over_ok=" + yesno(ok) + "|err=" + esc(err));
        stage.qsa_alloc = 1; stage.gdn_alloc = 7;
        ok = conversation_session_sizes(g, stage, junk, err);
        line("CARVE|gdn_over_ok=" + yesno(ok) + "|err=" + esc(err));
    }

    // every rejection the C++ test asserts
    reject([](auto& s) { s.kv.back().k.pop_back(); }, "late_draft_corruption");
    reject([](auto& s) { s.geometry[16] = 128; }, "expert_geometry");
    reject([](auto& s) { s.live.stage_parts.emplace_back(); }, "layer_split_live");
    reject([](auto& s) { s.checkpoints[0].stage_parts.emplace_back(); }, "layer_split_checkpoint");
    reject([](auto& s) { s.kv.pop_back(); }, "missing_kv_layer");
    reject([](auto& s) { s.live.gdn.pop_back(); }, "bad_live_recurrence");
    reject([](auto& s) { s.checkpoints.back().gdn.pop_back(); }, "bad_retained_checkpoint");
    reject([](auto& s) { s.checkpoints.back().ids[0] = 99; }, "foreign_checkpoint_prefix");
    reject([](auto& s) { s.checkpoints.back().imgs[0].hash++; }, "foreign_checkpoint_image");
    reject([](auto& s) { s.live.imgs[0].start = -1; }, "negative_image_position");
    reject([](auto& s) { s.live.ids[0] = -1; }, "negative_live_token");
    reject([](auto& s) { s.live.ple.push_back(0); }, "ple_mismatch");
    reject([](auto& s) { s.live.tails.push_back(0); }, "tail_mismatch");
    if (!zero_qsa) {
        reject([](auto& s) { s.kv[1].pooled.pop_back(); }, "indexer_spare_corruption");
        reject([](auto& s) { s.live.dead.pop_back(); }, "missing_spare_key");
        reject([](auto& s) { s.live.block_pos.pop_back(); }, "missing_indexer_metadata");
        ss.qsa_states = nullptr;
        std::string e2;
        ok = conversation_snapshot_validate(image, ss, g, draft.st, e2);
        line("MISSING_QSA|rejected=" + yesno(!ok) + "|err=" + esc(e2));
        ss.qsa_states = layers.data();
        reject([](auto& s) { s.layer_lo = 4; }, "other_layer_range");
    }

    // geometry arithmetic guards
    auto bad = g; bad.ssm_state_size = std::numeric_limits<int64_t>::max();
    sizes(bad, "overflow");
    bad = g; bad.qsa_interval = 0;
    sizes(bad, "zero_interval");
    bad = g; bad.n_head_kv = std::numeric_limits<int64_t>::max();
    line("KV_BYTES|overflow=" + std::to_string(conversation_kv_bytes(draft.st, bad, 9, false)));
    bad = g; bad.head_dim = std::numeric_limits<int64_t>::max();
    line("KV_BYTES|head_dim_overflow=" + std::to_string(conversation_kv_bytes(draft.st, bad, 9, false)));

    // ---------------------------------------------------------------- transfers
    register_regions();
    auto xfer = [&](const char* tag, int fc, int fs) {
        reset_state();
        fail_copy = fc; fail_sync = fs;
        std::string e;
        auto r = conversation_snapshot_restore(image, ss, g, draft.st, e);
        line(std::string("XFER|tag=") + tag + "|restore=" + restore_name(r) + "|err=" + esc(e) +
             "|copies=" + std::to_string(copy_calls) + "|syncs=" + std::to_string(sync_calls) +
             "|fingerprint=" + std::to_string(state_fingerprint()));
    };
    xfer("fail_sync_pre", 0, 1);
    xfer("fail_copy_first", 1, 0);
    xfer("fail_copy_partial", 2, 0);
    xfer("fail_sync_post", 0, 2);
    reset_state();
    {
        std::string e;
        auto r = conversation_snapshot_restore(image, ss, g, draft.st, e);
        bool gdn_match = gdn == image.live.gdn;
        bool ple_match = history == image.live.ple;
        line("XFER|tag=success|restore=" + restore_name(r) + "|err=" + esc(e) +
             "|copies=" + std::to_string(copy_calls) + "|gdn_match=" + yesno(gdn_match) +
             "|ple_match=" + yesno(ple_match) + "|ple_prev=" + std::to_string(ss.ple_prev[0]) + "," +
             std::to_string(ss.ple_prev[1]) + "|fingerprint=" + std::to_string(state_fingerprint()));
    }

    // read-back verification against the restored pools
    reset_state();
    {
        std::string e;
        auto rr = conversation_snapshot_restore(image, ss, g, draft.st, e);
        if (rr != ConversationRestore::restored)
            std::printf("VERIFY|restore_failed=%s\n", esc(e).c_str());
        uint64_t fp = 0;
        bool okv = conversation_kv_verify(image.kv.back(), draft.st, g, 9, false, fp, e);
        line("VERIFY|tag=draft|ok=" + yesno(okv) + "|err=" + esc(e) + "|fingerprint=" + std::to_string(fp));
        draft.data[0][0] ^= 1;
        uint64_t fp2 = 0;
        okv = conversation_kv_verify(image.kv.back(), draft.st, g, 9, false, fp2, e);
        line("VERIFY|tag=corrupt|ok=" + yesno(okv) + "|err=" + esc(e) + "|fingerprint=" + std::to_string(fp2));
        draft.data[0][0] ^= 1;
        fail_copy = copy_calls + 1;
        uint64_t fp3 = 0;
        okv = conversation_kv_verify(image.kv.back(), draft.st, g, 9, false, fp3, e);
        line("VERIFY|tag=copy_failed|ok=" + yesno(okv) + "|err=" + esc(e) + "|fingerprint=" + std::to_string(fp3));
    }

    // full and incremental capture, byte-compared
    reset_state();
    {
        std::string e;
        SavedConversation full;
        bool saved = conversation_snapshot_save(full, view, ss, g, draft.st, e);
        const size_t full_copies = copied_bytes;
        line("SAVE_FULL|ok=" + yesno(saved) + "|err=" + esc(e) + "|copies=" + std::to_string(full_copies) +
             "|bytes=" + std::to_string(full.bytes()));
        for (size_t i = 0; i < full.kv.size(); ++i)
            line("SAVE_FULL_KV|layer=" + std::to_string(i) + "|format=" + std::to_string(full.kv[i].format) +
                 "|cells=" + std::to_string(full.kv[i].cells) + "|heads=" + std::to_string(full.kv[i].heads) +
                 "|head_dim=" + std::to_string(full.kv[i].head_dim) +
                 "|page_size=" + std::to_string(full.kv[i].page_size) +
                 "|pooled_rows=" + std::to_string(full.kv[i].pooled_rows) +
                 "|idx_dim=" + std::to_string(full.kv[i].idx_dim) + "|k=" + std::to_string(fnv(flat(full.kv[i].k).data(), full.kv[i].k.size())) +
                 "|v=" + std::to_string(fnv(flat(full.kv[i].v).data(), full.kv[i].v.size())) +
                 "|ks=" + std::to_string(fnv(flat(full.kv[i].k_scale).data(), full.kv[i].k_scale.size())) +
                 "|vs=" + std::to_string(fnv(flat(full.kv[i].v_scale).data(), full.kv[i].v_scale.size())) +
                 "|pooled=" + std::to_string(fnv(flat(full.kv[i].pooled).data(), full.kv[i].pooled.size())));
        line("SAVE_FULL_STATE|gdn=" + std::to_string(fnv(full.live.gdn.data(), full.live.gdn.size())) +
             "|ple=" + std::to_string(fnv(full.live.ple.data(), full.live.ple.size())) +
             "|tails=" + std::to_string(fnv(full.live.tails.data(), full.live.tails.size())) +
             "|dead=" + std::to_string(fnv(full.live.dead.data(), full.live.dead.size())) +
             "|block_pos=" + std::to_string(fnv(full.live.block_pos.data(), full.live.block_pos.size())));

        ConversationKvReuse reuse{full.kv, 9, 9};
        size_t peak = 0;
        bool admitted = conversation_snapshot_capture_bytes(reuse, view, ss, g, draft.st, peak, e);
        line("REUSE_ADMISSION|ok=" + yesno(admitted) + "|err=" + esc(e) + "|peak=" + std::to_string(peak) +
             "|reuse_bytes=" + std::to_string(reuse.bytes()));

        copied_bytes = 0;
        size_t reused = 0;
        SavedConversation incremental;
        bool saved_inc = conversation_snapshot_save(incremental, view, ss, g, draft.st, e, std::move(reuse), &reused);
        line("SAVE_INCREMENTAL|ok=" + yesno(saved_inc) + "|err=" + esc(e) + "|reused=" + std::to_string(reused) +
             "|copies=" + std::to_string(copied_bytes) + "|sum=" +
             std::to_string(copied_bytes + reused) + "|full_copies=" + std::to_string(full_copies) +
             "|within_peak=" + yesno(incremental.bytes() <= peak) + "|bytes=" + std::to_string(incremental.bytes()));
        bool agree = incremental.kv.size() == full.kv.size();
        for (size_t i = 0; agree && i < full.kv.size(); ++i)
            agree = incremental.kv[i].k == full.kv[i].k && incremental.kv[i].v == full.kv[i].v &&
                    incremental.kv[i].k_scale == full.kv[i].k_scale &&
                    incremental.kv[i].v_scale == full.kv[i].v_scale &&
                    incremental.kv[i].pooled == full.kv[i].pooled;
        line("SAVE_INCREMENTAL|agree=" + yesno(agree) +
             "|gdn_equal=" + yesno(incremental.live.gdn == full.live.gdn));

        ConversationKvReuse bad_reuse{full.kv, 9, 9};
        bad_reuse.kv.back().k.pop_back();
        copy_calls = sync_calls = 0;
        size_t peak2 = 0;
        admitted = conversation_snapshot_capture_bytes(bad_reuse, view, ss, g, draft.st, peak2, e);
        line("REUSE_ADMISSION|malformed_ok=" + yesno(admitted) + "|err=" + esc(e) +
             "|copies=" + std::to_string(copy_calls) + "|syncs=" + std::to_string(sync_calls));

        ConversationKvReuse fail_reuse{full.kv, 9, 9};
        fail_copy = copy_calls + 1;
        SavedConversation unpublished;
        bool saved_fail = conversation_snapshot_save(unpublished, view, ss, g, draft.st, e, std::move(fail_reuse));
        line("SAVE_INCREMENTAL|copy_failure_ok=" + yesno(saved_fail) + "|err=" + esc(e) +
             "|kv_empty=" + yesno(unpublished.kv.empty()) +
             "|ids_empty=" + yesno(unpublished.live.ids.empty()));
    }
}

// ---------------------------------------------------------------- the buffer contract
// The segment ownership math is the admission math: `allocation_peak` is what a
// capture must be admitted against, and `segment_capacity` decides how a resize
// allocates. Both are observable, so both get golden lines.
void buffer_contract() {
    ConversationBuffer b;
    auto probe = [&](ConversationBuffer& buf, size_t next, const char* tag) {
        std::vector<uint8_t> got(buf.size(), 0);
        buf.read(got.data(), 0, buf.size());
        line(std::string("BUFFER|tag=") + tag + "|size=" + std::to_string(buf.size()) +
             "|bytes=" + std::to_string(buf.bytes()) +
             "|peak=" + std::to_string(buf.allocation_peak(next)) + "|next=" + std::to_string(next) +
             "|fill=" + std::to_string(fnv(got.data(), got.size())));
    };
    // a fresh capture allocates exactly its payload
    for (size_t n : {1uz, 1024uz, 65536uz, ConversationBuffer::segment_bytes - 1,
                     ConversationBuffer::segment_bytes, ConversationBuffer::segment_bytes + 1,
                     2 * ConversationBuffer::segment_bytes + 7}) {
        ConversationBuffer fresh;
        fresh.resize(n, 0x5a);
        probe(fresh, n, "fresh");
    }
    // later appends grow the last segment geometrically, bounded by 16 MiB
    {
        ConversationBuffer g;
        g.resize(4096, 0x11);
        probe(g, 4096, "append_start");
        for (int i = 0; i < 12; ++i) {
            g.resize(g.size() * 2 + 1000, 0x22);
            probe(g, g.size(), "append");
        }
        // and shrinking drops whole segments before trimming the last
        for (size_t n : {g.size() - 1, g.size() / 3, size_t(1), size_t(0), size_t(20) * ConversationBuffer::segment_bytes}) {
            g.resize(n, 0x33);
            probe(g, n, "shrink");
        }
    }
    // a failed admission must saturate, not wrap
    {
        ConversationBuffer h;
        h.resize(16, 0x44);
        line("BUFFER|tag=saturate|peak_max=" +
             std::to_string(h.allocation_peak(std::numeric_limits<size_t>::max()) == SIZE_MAX ? 1 : 0));
        // visit bounds, cross-segment read, equality across different segmentations
        std::vector<uint8_t> window(32, 0);
        line("BUFFER|tag=visit|in_range=" + yesno(h.read(window.data(), 4, 8)) +
             "|past_end=" + yesno(h.read(window.data(), 12, 8)) + "|zero=" + yesno(h.read(window.data(), 0, 0)));
        ConversationBuffer split;
        split.resize(ConversationBuffer::segment_bytes, 0x66);
        split.resize(ConversationBuffer::segment_bytes + 4096, 0x66);
        ConversationBuffer single;
        single.resize(ConversationBuffer::segment_bytes + 4096, 0x66);
        line("BUFFER|tag=segmentation|equal=" + yesno(split == single) +
             "|bytes_differ=" + yesno(split.bytes() != single.bytes()));
        ConversationBuffer shorter;
        shorter.resize(ConversationBuffer::segment_bytes + 4095, 0x66);
        line("BUFFER|tag=segmentation|shorter_equal=" + yesno(split == shorter));
        // pop_back across a segment boundary
        ConversationBuffer pop;
        pop.resize(ConversationBuffer::segment_bytes + 1, 0x77);
        pop.pop_back();
        probe(pop, pop.size(), "pop_to_boundary");
    }
}

}  // namespace

int main() {
    for (int format : {0, 1, 2})
        for (int experts : {256, 512})
            for (bool zero_qsa : {false, true})
                for (bool ple : {false, true}) {
                    World w(format, experts, zero_qsa, ple);
                    w.run();
                }
    buffer_contract();
    return 0;
}
