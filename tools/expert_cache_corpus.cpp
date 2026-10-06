// tools/expert_cache_corpus.cpp - the golden for expert_cache.cpp's deterministic core.
//
// expert_cache.cpp is the VRAM slot storage and the residency table. What is
// deterministic here is most of it: the profile file format, the learned-profile
// ranking, the allocation checks, the admission policy (global and per-layer), the
// segment arithmetic behind --vram-elastic, and the slot/verify paths. None of it
// needs a GPU - it needs a device that behaves, which is what the fake below is.
//
// Unlike the earlier corpora, this one DOES reach the device layer, so the fake is
// stateful: cudaMalloc hands back real memory, the copies really copy, and the VMM
// entry points really map and unmap. The rules are fixed and printed, so the Rust
// port can reproduce them:
//
//   * device memory is host memory; a copy is a memcpy.
//   * cuMemAddressReserve hands back a zeroed buffer; cuMemCreate a fresh zeroed
//     buffer behind a handle; cuMemMap copies the handle's bytes into the range and
//     cuMemUnmap copies them back before the handle is released - so a segment that
//     was unmapped comes back EMPTY, which is the behaviour the caller has to know.
//   * every scripted failure is a count or a flag; nothing is random.
//
// The HIP-only blocking-staging path (#ifdef STRATA_USE_HIP) is compiled out here and
// is NOT covered by this golden; neither is the gfx906 build, where the segmented
// cache refuses at open.
//
// Build (from the strata-rust checkout; $S is the Strata checkout):
//   ./tools/build_expert_cache_corpus.sh
//   ./target/corpus/eccorpus > crates/strata-core/tests/golden/expert_cache.txt

#include "strata/core/expert_cache.hpp"

#include <cuda.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct Fake {
    // runtime
    bool alloc_fail = false;
    bool memset_fail = false;
    bool info_ok = true;
    unsigned long long free_b = 64ull << 30, total_b = 64ull << 30;
    bool sync_ok = true, stream_sync_ok = true;
    unsigned long long fail_copy_bytes = 0;  // fail a copy of exactly this size
    cudaError_t last_error = cudaSuccess;
    // vmm
    bool vmm_supported = true, gran_ok = true, reserve_ok = true;
    size_t gran = 2u << 20;
    int fail_create_at = -1, fail_map_at = -1, fail_access_at = -1;  // the Nth call fails (1-based)
    // bookkeeping
    std::map<unsigned long long, std::pair<void*, size_t>> handles;
    std::map<unsigned long long, std::pair<unsigned long long, size_t>> maps;  // va -> (handle, size)
    unsigned long long next_handle = 1;
    int create_n = 0, map_n = 0, access_n = 0;
};

Fake g;

void reset() { g = Fake{}; }

const char* err_string(cudaError_t e) {
    switch (e) {
        case cudaSuccess: return "no error";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInvalidValue: return "invalid argument";
        default: return "unknown error";
    }
}

// ---- the fake device, reached through --wrap

void* alloc_zeroed(size_t n) {
    void* p = std::malloc(n);
    if (p) std::memset(p, 0, n);
    return p;
}

}  // namespace

extern "C" {

cudaError_t __wrap_cudaMalloc(void** p, size_t n) {
    if (g.alloc_fail) {
        g.last_error = cudaErrorMemoryAllocation;  // the real runtime leaves the error set for the caller to read
        return cudaErrorMemoryAllocation;
    }
    *p = alloc_zeroed(n);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
void __wrap_cudaFree(void* p) { std::free(p); }
cudaError_t __wrap_cudaMemset(void* p, int v, size_t n) {
    if (g.memset_fail) return cudaErrorMemoryAllocation;
    std::memset(p, v, n);
    return cudaSuccess;
}
cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (n == g.fail_copy_bytes) return cudaErrorMemoryAllocation;
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
    return __wrap_cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
}
cudaError_t __wrap_cudaDeviceSynchronize(void) { return g.sync_ok ? cudaSuccess : cudaErrorMemoryAllocation; }
cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t) { return g.stream_sync_ok ? cudaSuccess : cudaErrorMemoryAllocation; }
cudaError_t __wrap_cudaGetLastError(void) { return g.last_error; }
cudaError_t __wrap_cudaPeekAtLastError(void) { return g.last_error; }
cudaError_t __wrap_cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
const char* __wrap_cudaGetErrorString(cudaError_t e) { return err_string(e); }
cudaError_t __wrap_cudaMemGetInfo(size_t* f, size_t* t) {
    if (!g.info_ok) return cudaErrorMemoryAllocation;
    *f = (size_t) g.free_b;
    *t = (size_t) g.total_b;
    return cudaSuccess;
}

// ---- the fake VMM, reached through the driver entry-point lookup

cudaError_t cudaGetDriverEntryPointByVersion(const char* name, void** p, unsigned, cudaGetDriverEntryPointFlags,
                                             cudaDriverEntryPointQueryResult* q) {
    *q = cudaDriverEntryPointSuccess;
    *p = nullptr;
    if (!std::strcmp(name, "cuDeviceGet")) *p = (void*) +[](CUdevice* o, int i) { *o = i; return CUDA_SUCCESS; };
    else if (!std::strcmp(name, "cuDeviceGetAttribute"))
        *p = (void*) +[](int* o, CUdevice_attribute a, CUdevice) {
            // read per call, not memoized: this is the knob that varies per device and driver
            *o = (a == CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED && g.vmm_supported) ? 1 : 0;
            return CUDA_SUCCESS;
        };
    else if (!std::strcmp(name, "cuMemGetAllocationGranularity"))
        *p = (void*) +[](size_t* o, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) { return g.gran_ok ? (g.gran = g.gran ? g.gran : 0, *o = g.gran, CUDA_SUCCESS) : CUDA_ERROR_OUT_OF_MEMORY; };
    else if (!std::strcmp(name, "cuMemAddressReserve"))
        *p = (void*) +[](CUdeviceptr* o, size_t sz, size_t, CUdeviceptr, unsigned long long) { if (!g.reserve_ok) return CUDA_ERROR_OUT_OF_MEMORY; *o = (CUdeviceptr) alloc_zeroed(sz); return *o ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY; };
    else if (!std::strcmp(name, "cuMemAddressFree")) *p = (void*) +[](CUdeviceptr va, size_t) { std::free((void*) va); return CUDA_SUCCESS; };
    else if (!std::strcmp(name, "cuMemCreate"))
        *p = (void*) +[](CUmemGenericAllocationHandle* o, size_t sz, const CUmemAllocationProp*, unsigned long long) {
            if (++g.create_n == g.fail_create_at) return CUDA_ERROR_OUT_OF_MEMORY;
            void* b = alloc_zeroed(sz);
            if (!b) return CUDA_ERROR_OUT_OF_MEMORY;
            *o = g.next_handle++;
            g.handles[*o] = {b, sz};
            return CUDA_SUCCESS;
        };
    else if (!std::strcmp(name, "cuMemRelease"))
        *p = (void*) +[](CUmemGenericAllocationHandle h) { auto it = g.handles.find(h); if (it != g.handles.end()) std::free(it->second.first); g.handles.erase(it); return CUDA_SUCCESS; };
    else if (!std::strcmp(name, "cuMemMap"))
        *p = (void*) +[](CUdeviceptr va, size_t sz, size_t off, CUmemGenericAllocationHandle h, unsigned long long) {
            if (++g.map_n == g.fail_map_at) return CUDA_ERROR_OUT_OF_MEMORY;
            auto it = g.handles.find(h);
            if (it == g.handles.end()) return CUDA_ERROR_INVALID_VALUE;
            std::memcpy((void*) va, (char*) it->second.first + off, sz);  // the physical's bytes land in the range
            g.maps[va] = {h, sz};
            return CUDA_SUCCESS;
        };
    else if (!std::strcmp(name, "cuMemUnmap"))
        *p = (void*) +[](CUdeviceptr va, size_t sz) {
            auto it = g.maps.find(va);
            if (it != g.maps.end()) {
                auto h = g.handles.find(it->second.second);
                if (h != g.handles.end()) std::memcpy(h->second.first, (void*) va, sz);  // and go back the same way
                g.maps.erase(it);
            }
            return CUDA_SUCCESS;
        };
    else if (!std::strcmp(name, "cuMemSetAccess"))
        *p = (void*) +[](CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) { return ++g.access_n == g.fail_access_at ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS; };
    else *q = cudaDriverEntryPointSymbolNotFound;
    return *p ? cudaSuccess : cudaErrorInvalidValue;
}

}  // extern "C"

// ---- the golden

using strata::core::ExpertCache;
using strata::core::kNotResident;

namespace {

void line(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void line(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::printf("CORPUS|");
    std::vprintf(fmt, ap);
    std::printf("\n");
    va_end(ap);
}

std::string s16(const std::vector<std::pair<int32_t, int32_t>>& v, size_t n) {
    std::string s;
    for (size_t i = 0; i < v.size() && i < n; ++i) {
        if (i) s += ",";
        s += std::to_string(v[i].first) + ":" + std::to_string(v[i].second);
    }
    if (v.size() > n) s += ",...";
    return s;
}

const char* kPath = "/tmp/strata_ec_corpus_profile.bin";

void write_file(const std::vector<unsigned char>& b, const char* path) {
    std::FILE* f = std::fopen(path, "wb");
    std::fwrite(b.data(), 1, b.size(), f);
    std::fclose(f);
}

// the format, built by hand so the reader is tested against bytes rather than against itself
std::vector<unsigned char> make_profile(uint32_t nl, uint32_t ne, uint32_t slots,
                                        const std::vector<std::pair<uint16_t, uint16_t>>& ranked,
                                        const std::vector<int32_t>& table, const char* magic = "STRP") {
    std::vector<unsigned char> b;
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((unsigned char) (v >> (8 * i))); };
    auto u16 = [&](uint16_t v) { for (int i = 0; i < 2; ++i) b.push_back((unsigned char) (v >> (8 * i))); };
    auto i32 = [&](int32_t v) { u32((uint32_t) v); };
    b.push_back(magic[0]); b.push_back(magic[1]); b.push_back(magic[2]); b.push_back(magic[3]);
    u32(1); u32(nl); u32(ne); u32(slots); u32((uint32_t) ranked.size());
    for (auto& p : ranked) { u16(p.first); u16(p.second); }
    for (auto t : table) i32(t);
    return b;
}

void scenario_profile() {
    std::vector<std::pair<int32_t, int32_t>> ranked;
    int64_t slots = 0;
    std::string err;

    // a good file: 2 layers x 3 experts, 4 slots, 2 ranked
    write_file(make_profile(2, 3, 4, {{1, 2}, {0, 0}}, {-1, -1, -1, 1, -1, 0}), kPath);
    ranked.clear();
    if (strata::core::read_expert_profile(kPath, 2, 3, ranked, slots, err))
        line("PROFILE|read|%s|%lld", s16(ranked, 8).c_str(), (long long) slots);
    else
        line("PROFILE|read|FAIL|%s", err.c_str());

    write_file(make_profile(2, 3, 4, {}, {}, "XSTR"), kPath);
    err.clear();
    if (!strata::core::read_expert_profile(kPath, 2, 3, ranked, slots, err)) line("PROFILE|magic|%s", err.c_str());

    // too short to hold even a header
    write_file(std::vector<unsigned char>({'S', 'T', 'R', 'P', 1, 0, 0, 0}), kPath);
    err.clear();
    if (!strata::core::read_expert_profile(kPath, 2, 3, ranked, slots, err)) line("PROFILE|short|%s", err.c_str());

    write_file(make_profile(3, 3, 4, {}, {}), kPath);
    err.clear();
    if (!strata::core::read_expert_profile(kPath, 2, 3, ranked, slots, err)) line("PROFILE|dims|%s", err.c_str());

    write_file(make_profile(2, 3, 1, {{0, 0}, {1, 2}}, {}), kPath);
    err.clear();
    if (!strata::core::read_expert_profile(kPath, 2, 3, ranked, slots, err)) line("PROFILE|toomany|%s", err.c_str());

    // header says 2 pairs, only one present
    auto trunc = make_profile(2, 3, 4, {{0, 0}, {1, 2}}, {});
    trunc.resize(trunc.size() - 2);
    write_file(trunc, kPath);
    err.clear();
    if (!strata::core::read_expert_profile(kPath, 2, 3, ranked, slots, err)) line("PROFILE|truncated|%s", err.c_str());

    write_file(make_profile(2, 3, 4, {{0, 0}, {5, 2}}, {}), kPath);
    err.clear();
    if (!strata::core::read_expert_profile(kPath, 2, 3, ranked, slots, err)) line("PROFILE|range|%s", err.c_str());

    write_file(make_profile(2, 3, 4, {}, {}), kPath);
    ranked.clear();
    if (strata::core::read_expert_profile(kPath, 2, 3, ranked, slots, err))
        line("PROFILE|empty|%zu|%lld", ranked.size(), (long long) slots);

    // the writer, and what it refuses
    err.clear();
    std::vector<std::pair<int32_t, int32_t>> in = {{1, 2}, {0, 0}};
    if (strata::core::write_expert_profile(kPath, 2, 3, in, err)) {
        std::FILE* f = std::fopen(kPath, "rb");
        std::vector<unsigned char> got;
        int c;
        while ((c = std::fgetc(f)) != EOF) got.push_back((unsigned char) c);
        std::fclose(f);
        std::string hex;
        for (size_t i = 0; i < 28 && i < got.size(); ++i) { char b[8]; std::snprintf(b, sizeof b, "%02x", got[i]); hex += b; }
        line("PROFILE|write|%zu|%s", got.size(), hex.c_str());
        // and read it back through the real reader
        std::vector<std::pair<int32_t, int32_t>> back;
        int64_t bs = 0;
        std::string e2;
        if (strata::core::read_expert_profile(kPath, 2, 3, back, bs, e2))
            line("PROFILE|roundtrip|%s|%lld", s16(back, 8).c_str(), (long long) bs);
        else
            line("PROFILE|roundtrip|FAIL|%s", e2.c_str());
    } else {
        line("PROFILE|write|FAIL|%s", err.c_str());
    }
    err.clear();
    std::vector<std::pair<int32_t, int32_t>> bad = {{7, 0}};
    if (!strata::core::write_expert_profile(kPath, 2, 3, bad, err)) line("PROFILE|write_range|%s", err.c_str());
    err.clear();
    if (!strata::core::write_expert_profile(kPath, 0, 3, in, err)) line("PROFILE|write_fit|%s", err.c_str());
    std::remove(kPath);
    std::string p2 = std::string(kPath) + ".tmp";
    std::remove(p2.c_str());
}

void scenario_rank() {
    // 2 x 3. resident: 0,1 and 1,2. heat: 5 at 0,2 and 1 at 1,0. prior: 1,1 then 0,0.
    std::vector<uint8_t> res = {0, 1, 0, 0, 0, 1};
    std::vector<double> heat = {0, 0, 5, 1, 0, 0};
    std::vector<std::pair<int32_t, int32_t>> prior = {{1, 1}, {0, 0}, {9, 9}, {1, 1}};
    line("RANK|both|%s", s16(strata::core::rank_learned_profile(2, 3, res, heat, prior), 6).c_str());
    line("RANK|noheat|%s", s16(strata::core::rank_learned_profile(2, 3, res, {}, prior), 6).c_str());
    line("RANK|noprior|%s", s16(strata::core::rank_learned_profile(2, 3, res, heat, {}), 6).c_str());
    line("RANK|empty|%s", s16(strata::core::rank_learned_profile(2, 3, {}, {}, {}), 6).c_str());
    // all equal: index order decides
    std::vector<uint8_t> none = {0, 0, 0, 0, 0, 0};
    line("RANK|ties|%s", s16(strata::core::rank_learned_profile(2, 3, none, {}, {}), 6).c_str());
    // a short resident vector: past its end is "not resident", heat 0
    std::vector<uint8_t> shortres = {1, 0};
    line("RANK|shortvec|%s", s16(strata::core::rank_learned_profile(2, 3, shortres, {}, {}), 6).c_str());
}

void line_va(const char* label, ExpertCache& c) {
    line("%s|slots=%lld|full=%lld|bytes=%lld|full_bytes=%lld|gib=%.4f|resident=%lld|seg=%d|segbytes=%lld|mapped=%lld",
         label, (long long) c.slots(), (long long) c.full_slots(), (long long) c.bytes(),
         (long long) c.full_bytes(), c.gib(), (long long) c.resident(), c.segmented() ? 1 : 0,
         (long long) c.segment_bytes(), (long long) c.mapped_bytes());
}

void scenario_open() {
    std::string err;
    ExpertCache c;
    struct Case { const char* label; int64_t slots, nl, ne, blob; };
    const Case cases[] = {
        {"slots0", 0, 48, 512, 1310720}, {"slotsneg", -1, 48, 512, 1310720}, {"layers0", 4096, 0, 512, 1310720},
        {"expert0", 4096, 48, 0, 1310720}, {"blob0", 4096, 48, 512, 0},
    };
    for (const auto& k : cases) {
        err.clear();
        const bool ok = c.open(k.slots, k.nl, k.ne, k.blob, err);
        line("OPEN|refuse|%s|%d|%s", k.label, ok ? 1 : 0, err.c_str());
    }

    // the allocation is checked against the card before it is made
    reset();
    g.free_b = 4ull << 30;
    g.total_b = 24ull << 30;
    err.clear();
    if (!c.open(8192, 48, 512, 1310720, err)) line("OPEN|vram|%s", err.c_str());

    reset();
    g.info_ok = false;  // the card will not answer: the check is skipped and the allocation is attempted
    g.alloc_fail = true;
    err.clear();
    if (!c.open(8, 4, 8, 1024, err)) line("OPEN|alloc|%s", err.c_str());

    reset();
    g.info_ok = false;
    g.alloc_fail = false;
    g.memset_fail = false;
    if (c.open(8, 4, 8, 1024, err)) line("OPEN|infofail|the check is skipped and the allocation is made");
    c.close();

    reset();
    g.info_ok = false;
    g.memset_fail = true;
    err.clear();
    if (!c.open(8, 4, 8, 1024, err)) line("OPEN|memset|%s", err.c_str());

    reset();
    err.clear();
    if (c.open(64, 48, 512, 1310720, err)) {
        line_va("OPEN|ok", c);
        line("OPEN|within|%lld|%lld|%lld|%lld", (long long) c.slots_within(1310720),
             (long long) c.slots_within(2621440), (long long) c.slots_within(83886080), (long long) c.slots_within(0));
        line("OPEN|bytesof|%lld|%lld|%lld|%lld", (long long) c.bytes_of(0), (long long) c.bytes_of(3),
             (long long) c.bytes_of(99), (long long) c.bytes_of(-1));
    } else {
        line("OPEN|ok|FAIL|%s", err.c_str());
    }
    c.close();

    // sized slots: 256-byte aligned, each slot keeps its own size
    reset();
    err.clear();
    std::vector<int64_t> sizes = {1000, 2000, 256, 0};
    if (c.open_sized(sizes, 2, 4, err)) {
        line_va("SIZED|ok", c);
        line("SIZED|within|%lld|%lld|%lld", (long long) c.slots_within(1024), (long long) c.slots_within(2304),
             (long long) c.slots_within(4352));
        line("SIZED|bytesof|%lld|%lld|%lld", (long long) c.bytes_of(0), (long long) c.bytes_of(2),
             (long long) c.bytes_of(4));
    } else {
        line("SIZED|ok|FAIL|%s", err.c_str());
    }
    err.clear();
    if (!c.open_sized({}, 2, 4, err)) line("SIZED|refuse|%s", err.c_str());
    c.close();
}

void scenario_segmented() {
    std::string err;
    ExpertCache c;
    reset();
    g.gran = 4096;
    c.set_segment_bytes(16384);
    if (c.open(16, 2, 4, 4096, err)) {   // 64 KiB arena, 4 segments of 16 KiB
        line_va("SEG|open", c);
        line("SEG|within|%lld|%lld|%lld", (long long) c.slots_within(16384), (long long) c.slots_within(36864),
             (long long) c.slots_within(65536));
        // shrink rounds UP to a segment: keeping 20000 bytes keeps 2 segments
        err.clear();
        if (c.shrink(20000, err)) line_va("SEG|shrink20000", c);
        else line("SEG|shrink20000|FAIL|%s", err.c_str());
        // and grow rounds DOWN: wanting 36000 gets 2 segments, which is what it has
        err.clear();
        if (c.grow(36000, err)) line_va("SEG|grow36000", c);
        else line("SEG|grow36000|FAIL|%s", err.c_str());
        err.clear();
        if (c.grow(65536, err)) line_va("SEG|growall", c);
        else line("SEG|growall|FAIL|%s", err.c_str());
    } else {
        line("SEG|open|FAIL|%s", err.c_str());
    }
    c.close();

    // close() does NOT reset the segment request, so this cache is still segmented and
    // the refusal below is unreachable for it. A genuinely unsegmented cache refuses.
    reset();
    line_va("SEG|reopened", c);
    ExpertCache plain;
    if (plain.open(4, 2, 4, 1024, err)) {
        err.clear();
        if (!plain.shrink(1024, err)) line("SEG|shrink_unseg|%s", err.c_str());
        err.clear();
        if (!plain.grow(4096, err)) line("SEG|grow_unseg|%s", err.c_str());
    }
    plain.close();

    // the driver cannot back the third segment: the message names it, and what was mapped stays
    reset();
    g.gran = 4096;
    g.fail_create_at = 3;
    c.set_segment_bytes(16384);
    err.clear();
    if (!c.open(16, 2, 4, 4096, err)) line("SEG|createfail|%s", err.c_str());
    else line("SEG|createfail|unexpectedly ok");
    c.close();

    reset();
    g.gran = 4096;
    g.vmm_supported = false;
    c.set_segment_bytes(16384);
    err.clear();
    if (!c.open(16, 2, 4, 4096, err)) line("SEG|nosupport|%s", err.c_str());
    c.close();

    reset();
    g.gran = 4096;
    g.gran_ok = false;
    c.set_segment_bytes(16384);
    err.clear();
    if (!c.open(16, 2, 4, 4096, err)) line("SEG|granfail|%s", err.c_str());
    c.close();

    // what stays mapped keeps its bytes; what is unmapped and mapped again comes back as fresh
    // physical memory, because the driver hands out a new allocation, not the old one.
    reset();
    g.gran = 4096;
    c.set_segment_bytes(16384);
    if (c.open(16, 2, 4, 4096, err)) {
        unsigned char host[4096];
        for (int i = 0; i < 4096; ++i) host[i] = (unsigned char) (i * 7 + 1);
        std::vector<unsigned char> zeros(4096, 0);
        err.clear();
        const bool f2 = c.fill_slot(2, host, nullptr, err, 0);   // segment 0: the shrink keeps this one
        const bool f12 = c.fill_slot(12, host, nullptr, err, 0);  // segment 3: the shrink takes this one
        line("SEG|fill|%d|%d|%d", f2 ? 1 : 0, f12 ? 1 : 0, (int) c.fills());
        err.clear();
        const bool v2 = c.verify_slot(2, host, err, 0);
        const bool v12 = c.verify_slot(12, host, err, 0);
        line("SEG|verify|%d|%d", v2 ? 1 : 0, v12 ? 1 : 0);
        // Nothing is read while a segment is unmapped - the address range has no backing then,
        // and what a read would return is the driver's business, not the cache's. What the corpus
        // pins is that after a regrow the segment is NEW physical memory, not the old bytes.
        err.clear();
        if (c.shrink(16384, err)) {   // segments [0,1) stay
            line_va("SEG|shrunk", c);
            err.clear();
            if (c.grow(65536, err)) {
                line_va("SEG|regrown", c);
                err.clear();
                const bool a = c.verify_slot(2, host, err, 0);
                const bool b = c.verify_slot(12, zeros.data(), err, 0);
                line("SEG|verifyregrown|%d|%d", a ? 1 : 0, b ? 1 : 0);
            }
        }
        c.close();
    }
}

void scenario_admission() {
    std::string err;
    ExpertCache c;
    reset();
    if (!c.open(6, 3, 4, 1024, err)) { line("ADM|open failed"); return; }

    // the global path: arrival order, no eviction. Every admit is sequenced on its own line -
    // printf argument order is unspecified and arrival order is the thing under test.
    {
        const int a = c.admit(0, 0), b = c.admit(0, 1), d2 = c.admit(1, 3), e = c.admit(0, 0), f = c.admit(2, 2);
        line("ADM|global|%d|%d|%d|%d|%d", a, b, d2, e, f);
        const int g = c.admit(2, 0), h = c.admit(1, 0), i = c.admit(1, 1), j = c.admit(2, 3);
        line("ADM|globalfill|%d|%d|%d|%d", g, h, i, j);
        const int k = c.admit(2, 1), l = c.admit(0, 2);
        line("ADM|globalfull|%d|%d", k, l);
        line("ADM|globalres|%lld|%lld", (long long) c.resident(), (long long) c.full_slots());
        const int m = c.admit(3, 0), n = c.admit(0, 4);
        line("ADM|globaloob|%d|%d|%d", m, n, c.slot_of(3, 0));

        // replace moves the slot and leaves the old expert unresident
        // a normal replace moves the slot; a self-replace CLEARS it, because `old` is a reference
        // into the array and the second write lands on the same element the first read it from
        const int o = c.slot_of(0, 0);
        c.replace(0, 0, 2);
        line("ADM|replace|%d|%d|%d", o, c.slot_of(0, 2), c.slot_of(0, 0));
        const int p = c.slot_of(0, 2);
        c.replace(0, 2, 2);
        line("ADM|selfreplace|%d|%d", p, c.slot_of(0, 2));
    }

    // per-layer: 6 slots over 3 layers is q=2, the last layer takes the remainder
    ExpertCache d;
    reset();
    if (!d.open(7, 3, 4, 1024, err)) { line("ADM|open2 failed"); return; }
    d.set_per_layer_admission(true);
    long lo, hi;
    for (int l = 0; l < 3; ++l) {
        d.layer_slot_range(l, lo, hi);
        line("ADM|range|%d|%ld|%ld", l, lo, hi);
    }
    d.layer_slot_range(9, lo, hi);
    line("ADM|range_oob|%ld|%ld", lo, hi);
    {
        const int a = d.admit(0, 0), b = d.admit(0, 1), e = d.admit(0, 2);
        line("ADM|perlayer|%d|%d|%d", a, b, e);
        line("ADM|perlayerfull|%d", d.admit(0, 3));
        const int f = d.admit(1, 0), g = d.admit(2, 0);
        line("ADM|otherlayer|%d|%d", f, g);
        const int h = d.admit(2, 1), i = d.admit(2, 2), j = d.admit(2, 3);
        line("ADM|lastlayer|%d|%d|%d", h, i, j);
        line("ADM|perlayerres|%lld|%lld", (long long) d.resident(), (long long) d.full_slots());
    }
    c.close();
    d.close();
}

void scenario_copy() {
    std::string err;
    ExpertCache c;
    reset();
    if (!c.open(4, 2, 4, 1024, err)) { line("COPY|open failed"); return; }
    std::vector<unsigned char> host(1024);
    for (int i = 0; i < 1024; ++i) host[i] = (unsigned char) (i * 31 + 5);

    err.clear();
    if (c.fill_slot(0, host.data(), nullptr, err, 0)) line("COPY|fill0|%d", (int) c.fills());
    else line("COPY|fill0|FAIL|%s", err.c_str());
    err.clear();
    if (c.verify_slot(0, host.data(), err, 0)) line("COPY|verify0|ok");
    else line("COPY|verify0|FAIL|%s", err.c_str());

    // a partial blob: bytes < the slot, so only that many are compared
    err.clear();
    if (c.fill_slot(1, host.data(), nullptr, err, 256)) line("COPY|fill1partial|%d", (int) c.fills());
    host[200] ^= 0xFF;
    err.clear();
    if (!c.verify_slot(1, host.data(), err, 256)) line("COPY|verifydiff|%s", err.c_str());
    host[200] ^= 0xFF;
    err.clear();
    if (c.verify_slot(1, host.data(), err, 0)) line("COPY|verifyfull|ok");   // the byte the copy never wrote

    // bytes larger than the slot falls back to the slot size
    err.clear();
    if (c.fill_slot(2, host.data(), nullptr, err, 9999)) line("COPY|fill2big|%d", (int) c.fills());
    err.clear();
    if (!c.fill_slot(9, host.data(), nullptr, err, 0)) line("COPY|oob|%s", err.c_str());
    err.clear();
    if (!c.fill_slot(3, nullptr, nullptr, err, 0)) line("COPY|null|%s", err.c_str());

    // the blocking and queued forms, and their own messages
    err.clear();
    if (c.fill_slot_blocking(3, host.data(), err, 0)) line("COPY|block3|%d", (int) c.fills());
    else line("COPY|block3|FAIL|%s", err.c_str());
    err.clear();
    if (!c.fill_slot_blocking(9, host.data(), err, 0)) line("COPY|blockoob|%s", err.c_str());
    err.clear();
    if (!c.fill_slot_queued(9, host.data(), err, 0)) line("COPY|queueoob|%s", err.c_str());
    err.clear();
    if (!c.fill_slot_queued(3, nullptr, err, 0)) line("COPY|queuenull|%s", err.c_str());

    // a copy that fails says so, with the runtime's own words
    reset();
    g.fail_copy_bytes = 1024;
    err.clear();
    if (!c.fill_slot(0, host.data(), nullptr, err, 0)) line("COPY|copyfail|%s", err.c_str());
    err.clear();
    if (c.sync_queued(err)) line("COPY|syncok|a failed copy does not poison the queue");
    g.stream_sync_ok = false;
    err.clear();
    if (!c.sync_queued(err)) line("COPY|syncfail|%s", err.c_str());

    // verify reads back what it is told to compare, so a wrong slot is caught
    reset();
    if (c.open(4, 2, 4, 1024, err)) {
        std::vector<unsigned char> other = host;
        other[7] = 0;
        err.clear();
        if (!c.verify_slot(0, other.data(), err, 0)) line("COPY|verifyzero|%s", err.c_str());
        line("COPY|slotbounds|%d|%d|%d", c.device_slot(0) != nullptr, c.device_slot(4) != nullptr,
             c.device_slot(-1) != nullptr);
    }
    c.close();
}

void scenario_close() {
    std::string err;
    ExpertCache c;
    reset();
    if (!c.open(6, 3, 4, 1024, err)) { line("CLOSE|open failed"); return; }
    c.set_per_layer_admission(true);
    c.admit(0, 0);
    c.fill_slot(0, nullptr, nullptr, err, 0);  // fails: null host, no counter moved
    c.fill_slot(1, reinterpret_cast<const unsigned char*>(err.data()), nullptr, err, 0);
    line("CLOSE|before|%lld|%lld|%d", (long long) c.resident(), (long long) c.fills(), c.slot_of(0, 0));
    c.close();
    line("CLOSE|after|%lld|%lld|%lld|%lld|%d|%lld", (long long) c.slots(), (long long) c.full_slots(),
         (long long) c.resident(), (long long) c.fills(), c.slot_of(0, 0), (long long) c.segment_bytes());
    // and it is reusable
    c.set_segment_bytes(0);
    c.set_per_layer_admission(false);
    if (c.open(2, 1, 2, 512, err)) line("CLOSE|reopen|%lld|%lld", (long long) c.bytes(), (long long) c.resident());
    c.close();
}

}  // namespace

int main() {
    scenario_profile();
    scenario_rank();
    scenario_open();
    scenario_segmented();
    scenario_admission();
    scenario_copy();
    scenario_close();
    return 0;
}
