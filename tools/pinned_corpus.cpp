// tools/pinned_corpus.cpp - the golden for pinned.cu's deterministic core.
//
// pinned.cu is the platform boundary (mmap, madvise, hugepages, cudaHostRegister,
// the working-set lock), but a third of it is deterministic: the bounds list, the
// pin-cap decision, and the read plan with its per-layer FNV-1a checksums and its
// refusal messages. None of those reach a CUDA call, so the file compiles host-only
// against the stub headers in tools/cudastub (g++ needs -x c++ for a .cu) and the
// harness calls the real functions.
//
// The timing fields (seconds, read_seconds, copy_seconds) are NOT in the golden -
// they are what the machine was doing, not what it decided. Everything else is.
//
// Build (from the strata-rust checkout; $S is the Strata checkout):
//   S=/home/ramishra/work/LLM/Strata
//   R=/home/ramishra/work/LLM/strata-rust
//   g++ -std=c++20 -w -x c++ -DGGML_COMMON_DECL_C -I$S/include -I$S/third_party/ggml \
//       -I$R/tools/cudastub -c $S/src/core/pinned.cu -o $R/target/corpus/pinned.o
//   g++ -std=c++20 -w -I$S/include -I$S/third_party/ggml -I$R/tools/cudastub \
//       -I$R/tools -c $R/tools/pinned_corpus.cpp -o $R/target/corpus/pc.o
//   g++ -o target/corpus/pccorpus $R/target/corpus/pc.o $R/target/corpus/pinned.o \
//       $R/target/corpus/cstub.o $R/target/corpus/pstub.o
//
// The fixture files it writes are embedded in the golden as hex, so the Rust replay
// rebuilds them byte for byte. Paths never reach the output, so nothing needs
// normalising.
//
// The Windows branches (#ifdef _WIN32: sliced_pin_limit, the unbuffered ReadFile
// path, the DXGI budget) are compiled out here and are NOT covered by this golden.

#include "strata/core/pinned.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::core;

namespace {

fs::path g_root;
std::map<std::string, std::string> g_files;

void put(const std::string& rel, const std::string& bytes) {
    const fs::path p = g_root / rel;
    if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
    std::FILE* f = std::fopen(p.c_str(), "wb");
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    g_files[rel] = bytes;
}

// deterministic content so the Rust can rebuild the same bytes from the golden
std::string content(size_t n, uint64_t seed) {
    std::string s;
    s.resize(n);
    for (size_t i = 0; i < n; ++i) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        s[i] = (char) (unsigned char) (seed >> 33);
    }
    return s;
}

std::string hex(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        char b[3];
        std::snprintf(b, sizeof b, "%02x", c);
        out += b;
    }
    return out;
}

// ------------------------------------------------------------------ the pin cap

void cap_case(const char* tag, const char* value) {
    if (value == nullptr) {
        unsetenv("STRATA_ARENA_PIN_GIB");
    } else {
        setenv("STRATA_ARENA_PIN_GIB", value, 1);
    }
    std::printf("CORPUS|CAP|%s|value=%d\n", tag, arena_pin_cap_gib());
}

// the whole destination buffer, hashed - the golden records the hash, not 4 KB of hex
std::string hash_of(const std::vector<uint8_t>& v) {
    char b[32];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) fnv1a64(v.data(), v.size(), 1469598103934665603ull));
    return b;
}

// ------------------------------------------------------------------ the read plan

struct LoadCase {
    const char* tag;
    const char* file;              // fixture path, relative to the root
    std::vector<uint64_t> off;
    std::vector<uint64_t> bytes;
    int threads;
    uint64_t chunk;
    size_t dst_size;
};

void load_case(const LoadCase& c) {
    std::vector<uint8_t> dst(c.dst_size, 0);
    const LoadStats st = load_experts_ranges((g_root / c.file).string(), dst.data(), c.off, c.bytes, c.threads,
                                             c.chunk);
    std::printf("CORPUS|LOAD|%s|ok=%d|bytes=%llu|layers=%llu|error=%s|dst_hash=%s|checksums=", c.tag,
                st.ok ? 1 : 0, (unsigned long long) st.bytes, (unsigned long long) st.layers, st.error.c_str(),
                hash_of(dst).c_str());
    for (auto x : st.layer_checksums) std::printf("%llu,", (unsigned long long) x);
    std::printf("\n");
}

// ------------------------------------------------------------------ the direct loader

void direct_case(const char* tag, const char* file, const std::vector<uint64_t>& off,
                 const std::vector<uint64_t>& bytes, int threads, uint64_t chunk, size_t dst_size) {
    std::vector<uint8_t> dst(dst_size, 0);
    const LoadStats st = load_experts_direct((g_root / file).string(), dst.data(), off, bytes, threads, chunk);
    std::printf("CORPUS|DIRECT|%s|ok=%d|bytes=%llu|layers=%llu|error=%s|dst_hash=%s\n", tag, st.ok ? 1 : 0,
                (unsigned long long) st.bytes, (unsigned long long) st.layers, st.error.c_str(),
                hash_of(dst).c_str());
}

// ------------------------------------------------------------------ the unbuffered gate

void unbuf_case(const char* tag, const char* value) {
    if (value == nullptr) {
        unsetenv("STRATA_UNBUFFERED_LOAD");
    } else {
        setenv("STRATA_UNBUFFERED_LOAD", value, 1);
    }
    std::string why;
    const bool r = experts_unbuffered({}, 1 << 20, why, true, (uint64_t) -1);
    std::printf("CORPUS|UNBUF|%s|ret=%d|why=%s\n", tag, r ? 1 : 0, why.c_str());
}

}  // namespace

int main() {
    g_root = fs::temp_directory_path() / "strata-pinned-replay";
    fs::remove_all(g_root);
    fs::create_directories(g_root);

    // fixtures: three packs, one of them deliberately short
    put("pack_full.bin", content(3072, 1));
    put("pack_short.bin", content(1024, 1));  // same content, truncated by 2 KiB
    put("pack_gap.bin", content(4096, 7));

    // ---- arena_pin_cap_gib --------------------------------------------------
    cap_case("unset", nullptr);
    cap_case("empty", "");
    cap_case("auto", "auto");
    cap_case("five", "5");
    cap_case("zero", "0");
    cap_case("negative", "-3");
    cap_case("not_a_number", "abc");
    cap_case("leading_digits", "7x");
    cap_case("leading_space", " 9");

    // ---- uniform_bounds: it lives in an anonymous namespace inside pinned.cu,
    // so the harness cannot call it. The Rust covers it with hand-derived unit
    // tests instead - five lines, no I/O, and the tail push is the whole point.

    // ---- load_experts_ranges: the plan, the checksums, the refusals ---------
    load_case({"clean_1t", "pack_full.bin", {0, 1024, 2048}, {1024, 1024, 1024}, 1, 4096, 3072});
    load_case({"clean_4t", "pack_full.bin", {0, 1024, 2048}, {1024, 1024, 1024}, 4, 4096, 3072});
    load_case({"small_chunks", "pack_full.bin", {0, 1024, 2048}, {1024, 1024, 1024}, 3, 256, 3072});
    load_case({"ragged_chunk", "pack_full.bin", {0, 1024, 2048}, {1024, 1024, 1024}, 2, 700, 3072});
    load_case({"zero_layer", "pack_full.bin", {0, 1024, 2048}, {1024, 0, 1024}, 2, 512, 3072});
    load_case({"gap", "pack_gap.bin", {0, 2048}, {1024, 1024}, 2, 512, 4096});
    load_case({"short_read", "pack_short.bin", {0, 1024, 2048}, {1024, 1024, 1024}, 1, 4096, 3072});
    load_case({"seek_past_eof", "pack_short.bin", {0, 2048}, {512, 512}, 1, 256, 4096});
    load_case({"threads_zero", "pack_full.bin", {0, 1024}, {1024, 1024}, 0, 512, 3072});

    // ---- load_experts_direct (the Windows path; Linux returns the defaults) -
    direct_case("direct_clean", "pack_full.bin", {0, 1024, 2048}, {1024, 1024, 1024}, 2, 1024, 3072);
    direct_case("direct_unaligned", "pack_full.bin", {0, 1024, 2048}, {1024, 1024, 1024}, 2, 512, 3072);

    // ---- experts_unbuffered -------------------------------------------------
    unbuf_case("unset", nullptr);
    unbuf_case("empty", "");
    unbuf_case("one", "1");
    unbuf_case("zero", "0");
    unbuf_case("word", "auto");

    std::printf("\n");
    for (const auto& kv : g_files) std::printf("CORPUS|FILE|%s|%s\n", kv.first.c_str(), hex(kv.second).c_str());
    return 0;
}
