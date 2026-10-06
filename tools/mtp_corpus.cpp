// tools/mtp_corpus.cpp — drive MtpDrafter::load() and bind_bytes() over fixture dirs and print what they do.
//
// `load` is the deterministic core: it parses the runtime index, refuses on what it cannot find, decides the
// K/V ring, and carves the arena through a 256-byte bump allocator.  The CUDA calls around it are faked with
// real host memory (--wrap), so the layout is the real code's answer on this box.
//
// The size helpers below are TRANSCRIBED from their .cu files (each cites the line) because this corpus builds
// without nvcc.  A misread of one of those is invisible here.  Everything else is the real code.
//
// Build: see the bottom of the file.  stdout is the golden; stderr (the loader's own chatter) is discarded.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "strata/core/layout.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

// the corpus reads the drafter's private carve; the access hack needs every header above already in
#define private public
#include "strata/core/mtp.hpp"
#undef private

namespace strata::kernels {

// native_mmvq.cu:1250 (Q8K = 32, sizeof(Q81Block) = 36)
std::size_t native_q8_1_bytes(int n_in, int ncols) {
    return std::size_t(ncols) * std::size_t(n_in / 32) * 36;
}

// s2_expert_grouped.cu:568
uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) + ((xh + 15) & ~15ull);
}

// sampler.cu:1037,1048 (kSplitBlockSpan = 4096, kSplitMaxBlocks = 64, kSelMax = 64, sizeof(int2) = 8)
static int coupled_blocks(int nv) {
    return (nv + 4096 - 1) / 4096;
}
size_t coupled_draft_scratch_bytes(int nv) {
    if (nv <= 0 || coupled_blocks(nv) > 64) return 0;
    return (size_t) coupled_blocks(nv) * (size_t) 64 * 8;
}

// shared_expert.cu:234
uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}

// qsa_decode_attn.cu:490 (HD = 256, CHUNK = 64)
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + 64 - 1) / 64;
    return (uint64_t) chunks * (uint64_t) s.n_head * (256 + 2) + 64;
}

// kv_stream.cu:192
uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                          : rows * (uint64_t) s.head_dim * 2 * 2;
}

// the corpus never reaches these; they are here so the link is honest rather than ignored
const RopeScaling& rope_scaling() {
    static RopeScaling s{};
    return s;
}
void build_rope_table(int, double, int, float*, float*) {}
void build_rope_table(int, const RopeScaling&, int, float*, float*) {}
void rope_table_set(const float*, const float*, int, const RopeScaling&) {}
void kv_stream_reset(const KvStreamMap&, void*) {}
void kv_ring_table(int32_t*, int64_t, int64_t, void*) {}

}  // namespace strata::kernels

namespace strata::core {
bool peer_portable() {
    return false;
}
}  // namespace strata::core

#include <execinfo.h>
#include <csignal>
void segv(int) {
    void* bt[24];
    int n = backtrace(bt, 24);
    std::fprintf(stderr, "SEGFAULT backtrace:\n");
    backtrace_symbols_fd(bt, n, 2);
    std::abort();
}

extern "C" void strata_stub_report(const char* n) {
    std::fprintf(stderr, "STUB CALLED: %s\n", n);
    std::abort();
}

// ---------------------------------------------------------------- the CUDA seam

static std::vector<uint8_t> g_slab;
static size_t g_slab_used = 0;
static size_t g_host_used = 0;
static size_t g_slab_peak = 0, g_host_peak = 0;
static int g_fail_malloc_at = 0, g_malloc_calls = 0;
static std::vector<std::pair<size_t, size_t>> g_allocs;  // (offset in the slab, size)

static std::vector<uint8_t>& slab() {
    if (g_slab.empty()) g_slab.resize(3ull << 30);
    return g_slab;
}

extern "C" {
cudaError_t __wrap_cudaMalloc(void** p, size_t n) {
    ++g_malloc_calls;
    if (g_fail_malloc_at && g_malloc_calls == g_fail_malloc_at) return cudaErrorMemoryAllocation;
    auto& s = slab();
    if (g_slab_used + n > s.size()) return cudaErrorMemoryAllocation;
    *p = s.data() + g_slab_used;
    std::memset(*p, 0xa5, n);
    g_allocs.push_back({g_slab_used, n});
    g_slab_used += n;
    if (g_slab_used > g_slab_peak) g_slab_peak = g_slab_used;
    return cudaSuccess;
}
cudaError_t __wrap_cudaFree(void*) {
    return cudaSuccess;
}
cudaError_t __wrap_cudaHostAlloc(void** p, size_t n, unsigned) {
    auto& s = slab();
    if (g_host_used + n > s.size()) return cudaErrorMemoryAllocation;
    *p = s.data() + s.size() - (g_host_used += n);
    std::memset(*p, 0, n);
    if (g_host_used > g_host_peak) g_host_peak = g_host_used;
    return cudaSuccess;
}
cudaError_t __wrap_cudaFreeHost(void*) {
    return cudaSuccess;
}
cudaError_t __wrap_cudaHostGetDevicePointer(void** d, void* h, unsigned) {
    *d = h;
    return cudaSuccess;
}
cudaError_t __wrap_cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
    std::memcpy(d, s, n);
    return cudaSuccess;
}
cudaError_t __wrap_cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t) {
    std::memcpy(d, s, n);
    return cudaSuccess;
}
cudaError_t __wrap_cudaMemset(void* p, int v, size_t n) {
    std::memset(p, v, n);
    return cudaSuccess;
}
cudaError_t __wrap_cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t) {
    std::memset(p, v, n);
    return cudaSuccess;
}
cudaError_t __wrap_cudaDeviceSynchronize() {
    return cudaSuccess;
}
cudaError_t __wrap_cudaGetDevice(int* d) {
    *d = 0;
    return cudaSuccess;
}
cudaError_t __wrap_cudaSetDevice(int) {
    return cudaSuccess;
}
cudaError_t __wrap_cudaGetLastError() {
    return cudaSuccess;
}
cudaError_t __wrap_cudaMemGetInfo(size_t* f, size_t* t) {
    *f = 1ull << 30;
    *t = 1ull << 31;
    return cudaSuccess;
}
cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int) {
    *s = (cudaStream_t) (uintptr_t) 0x1;
    return cudaSuccess;
}
cudaError_t __wrap_cudaStreamDestroy(cudaStream_t) {
    return cudaSuccess;
}
const char* cudaGetErrorString(cudaError_t) {
    return "stub";
}
}

// ---------------------------------------------------------------- fixtures

static std::string g_root;

static void write_file(const std::string& path, const std::string& bytes) {
    std::ofstream o(path, std::ios::binary);
    o.write(bytes.data(), (std::streamsize) bytes.size());
}

static std::string blob(size_t n, uint8_t seed) {
    std::string s((size_t) n, '\0');
    for (size_t i = 0; i < s.size(); ++i) s[i] = (char) (uint8_t)(i * 31 + seed);
    return s;
}

// the nine q8_0 tensors `load` requires, plus one f32 and one bf16 so the lookups have something to find
static const char* IDX_OK =
    "fc_embedding.weight q8_0 0 0 0 1024\n"
    "fc_hidden.weight q8_0 0 1024 1024 1024\n"
    "self_attn.q_proj.weight q8_0 0 2048 2048 1024\n"
    "self_attn.k_proj.weight q8_0 0 3072 3072 1024\n"
    "self_attn.v_proj.weight q8_0 0 4096 4096 1024\n"
    "self_attn.o_proj.weight q8_0 0 5120 5120 1024\n"
    "mlp.shared_expert.gate_proj.weight q8_0 0 6144 6144 1024\n"
    "mlp.shared_expert.up_proj.weight q8_0 0 7168 7168 1024\n"
    "mlp.shared_expert.down_proj.weight q8_0 0 8192 8192 1024\n"
    "norm.weight f32 0 9216 9216 1024\n"
    "head_bf16.weight bf16 0 10240 10240 1024\n";

static void mk_fixture(const std::string& name, const char* idx, size_t blob_bytes, bool with_experts,
                       size_t experts_bytes) {
    std::string dir = g_root + "/" + name;
    std::string cmd = "mkdir -p " + dir;
    if (system(cmd.c_str()) != 0) std::abort();
    if (idx) write_file(dir + "/dense.txt", idx);
    write_file(dir + "/dense.bin", blob(blob_bytes, 7));
    if (with_experts) write_file(dir + "/experts.bin", blob(experts_bytes, 11));
}

// ---------------------------------------------------------------- observations

using strata::core::ModelGeometry;
using strata::core::SessionState;

static strata::core::QsaState g_qsa_state;

static ModelGeometry geom;
static SessionState ss;

static void show(const char* tag, const std::string& dir, int max_t, int64_t window, int fail_malloc_at,
                 int64_t resident = 0) {
    strata::core::qsa_set_kv_resident(resident);
    g_fail_malloc_at = fail_malloc_at;
    g_malloc_calls = 0;
    g_slab_used = 0;
    g_host_used = 0;
    g_allocs.clear();
    strata::core::MtpDrafter d;
    std::string err;
    std::fprintf(stderr, "[trace] load %s\n", dir.c_str());
    bool ok = d.load(g_root + "/" + dir, geom, ss, max_t, err, window);
    std::printf("LOAD|%s|ok=%d|err=%s|vram=%llu|kv_mode=%d\n", tag, ok, err.c_str(),
                (unsigned long long) d.vram_bytes(), (int) d.kv_state().kv_mode);
    for (size_t i = 0; i < g_allocs.size(); ++i)
        std::printf("  ALLOC|%zu|off=%llu|bytes=%llu\n", i, (unsigned long long) g_allocs[i].first,
                    (unsigned long long) g_allocs[i].second);
    if (ok) {
        auto off = [&](const void* p) {
            return (unsigned long long) ((const uint8_t*) p - (const uint8_t*) d.arena_);
        };
        std::printf("  FIELDS|tok=%llu|ident=%llu|Rin=%llu|ident_last=%llu|attn_scratch=%llu|sh_scratch=%llu|grp_counts=%llu|probs=%llu\n",
                    off(d.tok_), off(d.ident_), off(d.Rin_),
                    (unsigned long long) (((const int32_t*) d.ident_)[(size_t) d.max_t_ * (size_t) d.cap_ - 1]),
                    off(d.attn_scratch_), off(d.sh_scratch_), off(d.grp_counts_), off(d.probs_));
        std::printf("  BIND|head_row=1024|nv=151936|bytes=%llu\n",
                    (unsigned long long) d.bind_bytes(1024, 151936));
    }
}

int main(int argc, char** argv) {
    (void) argc;
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::signal(SIGSEGV, segv);
    g_root = argv[0];
    g_root += ".d";
    std::fprintf(stderr, "[trace] main start\n");
    std::string cmd = "rm -rf " + g_root;
    if (system(cmd.c_str()) != 0) std::abort();

    // the real widths; n_expert is 0 so the experts fixture stays small
    geom = ModelGeometry{};
    geom.n_expert = 0;
    // `load` reads `ss.qsa_states[primary].max_cells`; the session's array is normally carved by session_init.
    ss.qsa_states = &g_qsa_state;
    g_qsa_state.max_cells = 4096;
    ss.max_cells = 4096;
    ss.k = 10;

    mk_fixture("ok", IDX_OK, 11264, true, 0);
    mk_fixture("ok_vocab", IDX_OK, 11264, true, 0);
    mk_fixture("no_idx", nullptr, 11264, true, 0);
    mk_fixture("malformed", "norm.weight f32 0 0 0 1024\nshort line\n", 11264, true, 0);
    mk_fixture("missing_required",
               "norm.weight f32 0 0 0 1024\nnorm2.weight f32 0 1024 1024 1024\n", 2048, true, 0);
    mk_fixture("no_experts", IDX_OK, 11264, false, 0);
    write_file(g_root + "/ok_vocab/draft_vocab.bin", blob(4 * 64, 3));

    show("t4_res0_window0", "ok", 4, 0, 0, 0);
    show("t4_res0_window32", "ok", 4, 32, 0, 0);
    show("t4_res4096_window32", "ok", 4, 32, 0, 4096);
    show("t4_res4096_window0", "ok", 4, 0, 0, 4096);
    show("t4_res4096_window0_off", "ok", 4, 0, 0, 4096);
    show("t4_res100000_window32", "ok", 4, 32, 0, 100000);
    show("t4_res100000_window0", "ok", 4, 0, 0, 100000);
    show("t4_res100000_window4095", "ok", 4, 4095, 0, 100000);
    show("t4_res100000_window5000", "ok", 4, 5000, 0, 100000);
    show("t1_res100000_window32", "ok", 1, 32, 0, 100000);
    show("t8_res100000_window32", "ok", 8, 32, 0, 100000);
    show("t0", "ok", 0, 32, 0, 100000);
    show("t9", "ok", 9, 32, 0, 100000);
    show("no_idx", "no_idx", 4, 32, 0, 100000);
    show("malformed", "malformed", 4, 32, 0, 100000);
    show("missing_required", "missing_required", 4, 32, 0, 100000);
    show("no_experts", "no_experts", 4, 32, 0, 100000);
    show("malloc_fail_dense", "ok", 4, 32, 1, 100000);
    show("malloc_fail_state", "ok", 4, 32, 3, 100000);
    show("malloc_fail_arena", "ok", 4, 32, 4, 100000);

    // bind_bytes: the draft-vocab file and the coupled switch both add to it
    strata::core::set_coupled_draft(false);
    show("bind_coupled_off", "ok", 4, 0, 0, 100000);
    show("bind_vocab", "ok_vocab", 4, 0, 0, 100000);
    strata::core::set_coupled_draft(true);
    show("bind_coupled_on", "ok", 4, 0, 0, 100000);
    show("bind_vocab_coupled", "ok_vocab", 4, 0, 0, 100000);
    strata::core::set_coupled_draft(false);
        return 0;
}
