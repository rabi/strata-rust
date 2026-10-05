// tools/expert_plan_corpus.cpp - the golden for expert_source's pure policy.
//
// expert_source.cpp's admission gate and cache-complement planner are already
// pure functions with injectable inputs (the header says so: "kept CPU-only so
// selection and byte accounting can be tested without initializing a GPU"). This
// harness compiles the REAL expert_source.cpp on the host - no CUDA toolkit, the
// runtime calls are stubbed in cudastub_defs.cpp and none of them are reached by
// these functions - runs a matrix of cases through each, and prints one CORPUS
// line per observation. tests/expert_plan_corpus.rs replays the same matrix
// through the Rust port and the two line streams must be identical.
//
// Build (from the strata-rust checkout; $S is the Strata checkout):
//   S=/home/ramishra/work/LLM/Strata
//   R=/home/ramishra/work/LLM/strata-rust
//   g++ -std=c++20 -w -DGGML_COMMON_DECL_C -I$S/include -I$S/third_party/ggml \
//       -I$R/tools/cudastub -c $S/src/core/expert_source.cpp -o $R/target/corpus/es.o
//   g++ -std=c++20 -w -I$S/include -I$S/third_party/ggml -I$R/tools/cudastub \
//       -I$R/tools -c $R/tools/expert_plan_corpus.cpp -o $R/target/corpus/epc.o
//   g++ -std=c++20 -w -o $R/target/corpus/epcorpus $R/target/corpus/epc.o \
//       $R/target/corpus/es.o $R/target/corpus/stubs.o
//   ./target/corpus/epcorpus | <normalize the temp path to <TMP>> \
//       > crates/strata-core/tests/golden/expert_plan.txt
//
// The fixture files the harness writes (the fake /proc and cgroup tree) are
// embedded in the golden as CORPUS|BLOB/CORPUS|FILE lines, so the Rust replay can
// rebuild them byte for byte.

#include "strata/core/expert_source.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::core;
using namespace strata::core::detail;

namespace {

fs::path g_root;
// path relative to the root -> bytes, so the Rust replay can rebuild the tree.
std::map<std::string, std::string> g_files;

void put(const fs::path& rel, const std::string& bytes) {
    const fs::path abs = g_root / rel;
    if (!abs.parent_path().empty()) fs::create_directories(abs.parent_path());
    std::FILE* f = std::fopen(abs.c_str(), "wb");
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    g_files[rel.string()] = bytes;
}

// ------------------------------------------------------------------ cgroup math

void cgroup_case(const char* tag, uint64_t limit, bool valid, uint64_t current, uint64_t inactive,
                 uint64_t dirty, uint64_t writeback) {
    CgroupMemoryStat s;
    s.valid = valid;
    s.current = current;
    s.inactive_file = inactive;
    s.file_dirty = dirty;
    s.file_writeback = writeback;
    uint64_t bytes = 0;
    const bool ok = cgroup_available_bytes(limit, s, bytes);
    std::printf("CORPUS|CGROUP|%s|ok=%d|bytes=%llu\n", tag, ok ? 1 : 0, (unsigned long long) bytes);
}

// ------------------------------------------------------- the complement planner

struct Pair {
    int32_t a, b;
};

void plan_case(const char* tag, int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& blob_bytes,
               const std::vector<Pair>& primary, const std::vector<Pair>& additional, size_t show) {
    std::vector<std::pair<int32_t, int32_t>> p, a;
    for (const auto& x : primary) p.push_back({x.a, x.b});
    for (const auto& x : additional) a.push_back({x.a, x.b});
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string err;
    const bool ok = make_cache_complement_plan(n_layers, n_expert, blob_bytes, p, a, offsets, bytes, err);
    std::printf("CORPUS|PLAN|%s|ok=%d|err=%s|bytes=%llu|count=%zu|sentinels=%zu", tag, ok ? 1 : 0, err.c_str(),
                (unsigned long long) bytes, offsets.size(),
                (size_t) std::count(offsets.begin(), offsets.end(), kNoCacheComplement));
    // the first `show` offsets in index order, so the arithmetic is visible
    std::printf("|first=");
    for (size_t i = 0; i < show && i < offsets.size(); ++i) std::printf("%llu,", (unsigned long long) offsets[i]);
    std::printf("\n");
}

void exchange_case(const char* tag, std::vector<uint64_t> offsets, size_t in, size_t out) {
    std::vector<uint64_t> before = offsets;
    const bool ok = exchange_cache_complement(offsets, in, out);
    std::printf("CORPUS|EXCHANGE|%s|ok=%d|changed=%d", tag, ok ? 1 : 0, offsets != before ? 1 : 0);
    // which index holds which offset, for the indices that moved
    for (size_t i = 0; i < offsets.size() && i < 8; ++i) {
        if (offsets[i] != before[i]) std::printf("|%zu:%llu->%llu", i, (unsigned long long) before[i],
                                                 (unsigned long long) offsets[i]);
    }
    std::printf("\n");
}

void fallback_case(const char* tag, const std::vector<uint64_t>& offsets, size_t index, bool have_complement) {
    static std::vector<uint8_t> complement(4096, 0x5a);
    static std::vector<uint8_t> mapped(4096, 0xa5);
    const uint8_t* c = have_complement ? complement.data() : nullptr;
    const uint8_t* got = cache_complement_blob_or_fallback(index, offsets, c, mapped.data());
    // which backing it resolved to, and the offset into it
    if (got == mapped.data()) {
        std::printf("CORPUS|FALLBACK|%s|which=mapped|off=0\n", tag);
    } else if (got == c) {
        std::printf("CORPUS|FALLBACK|%s|which=complement|off=0\n", tag);
    } else {
        std::printf("CORPUS|FALLBACK|%s|which=complement|off=%llu\n", tag,
                    (unsigned long long) (got - c));
    }
}

// ------------------------------------------------------------- the resident keep

void keep_case(const char* tag, const std::vector<uint64_t>& slot_bytes, uint64_t base, uint64_t budget,
               int64_t lend_from) {
    std::printf("CORPUS|KEEP|%s|keep=%lld\n", tag,
                (long long) choose_resident_keep_from(slot_bytes, base, budget, lend_from));
}

// ------------------------------------------------------------- host RAM + cgroup

// Each case gets its own subtree under the root - the files are per-case state, and
// a case that omits a file must not inherit the previous case's.
void host_case(const char* tag, const std::map<std::string, std::string>& tree) {
    for (const auto& kv : tree) put(fs::path(tag) / kv.first, kv.second);
    const fs::path root = g_root / tag;
    HostMemory m;
    const bool ok =
        host_available_memory(m, (root / "cgroup_file").string(), (root / "groups").string(), root);
    std::printf("CORPUS|HOST|%s|ok=%d|available=%llu|cgroup_limit=%llu\n", tag, ok ? 1 : 0,
                (unsigned long long) m.available, (unsigned long long) m.cgroup_limit);
}

}  // namespace

int main() {
    g_root = fs::temp_directory_path() / "strata-ep-replay";
    fs::remove_all(g_root);
    fs::create_directories(g_root);
    fs::current_path(g_root);

    const uint64_t GiB = 1ull << 30;

    // ---- cgroup_available_bytes: the clean-reclaim arithmetic, every bound ----
    cgroup_case("invalid_stat", 100 * GiB, false, 50 * GiB, 20 * GiB, GiB, GiB);
    cgroup_case("plain", 100 * GiB, true, 50 * GiB, 20 * GiB, 5 * GiB, 2 * GiB);
    cgroup_case("dirty_eats_reclaim", 100 * GiB, true, 50 * GiB, 20 * GiB, 20 * GiB, 0);
    cgroup_case("writeback_eats_reclaim", 100 * GiB, true, 50 * GiB, 20 * GiB, 0, 20 * GiB);
    cgroup_case("inactive_over_charged", 100 * GiB, true, 10 * GiB, 20 * GiB, 0, 0);
    cgroup_case("over_limit_after_reclaim", 10 * GiB, true, 50 * GiB, 20 * GiB, 0, 0);
    cgroup_case("all_zero", 100 * GiB, true, 0, 0, 0, 0);
    cgroup_case("limit_zero", 0, true, 0, 0, 0, 0);
    cgroup_case("dirty_over_reclaim", 100 * GiB, true, 50 * GiB, 20 * GiB, 30 * GiB, 0);

    // ---- make_cache_complement_plan: selection and byte accounting ----------
    plan_case("no_pairs", 2, 3, {100, 200}, {}, {}, 12);
    plan_case("primary_marks", 2, 3, {100, 200}, {{0, 1}}, {}, 12);
    plan_case("additional_marks", 2, 3, {100, 200}, {}, {{1, 0}}, 12);
    plan_case("both_tiers", 2, 3, {100, 200}, {{0, 1}}, {{1, 0}}, 12);
    plan_case("overlap", 2, 3, {100, 200}, {{0, 1}}, {{0, 1}}, 0);
    plan_case("dup_primary", 2, 3, {100, 200}, {{0, 1}, {0, 1}}, {}, 0);
    plan_case("dup_additional", 2, 3, {100, 200}, {}, {{0, 1}, {0, 1}}, 0);
    plan_case("pair_outside", 2, 3, {100, 200}, {{2, 0}}, {}, 0);
    plan_case("pair_negative", 2, 3, {100, 200}, {{0, -1}}, {}, 0);
    plan_case("zero_blob", 2, 3, {100, 0}, {}, {}, 0);
    plan_case("bad_layers", 0, 3, {100}, {}, {}, 0);
    plan_case("bad_expert", 2, 0, {100, 200}, {}, {}, 0);
    plan_case("size_mismatch", 2, 3, {100}, {}, {}, 0);
    plan_case("huge_geometry", 1000000000, 1000000000, {1}, {}, {}, 0);
    plan_case("per_layer_sizes", 3, 2, {7, 11, 13}, {{1, 1}}, {}, 12);

    // ---- exchange_cache_complement: the adaptive tier's swap -----------------
    exchange_case("swap", {0, kNoCacheComplement, 100}, 2, 1);
    exchange_case("same_index", {0, 100}, 1, 1);
    exchange_case("in_not_in_copy", {kNoCacheComplement, 100}, 0, 1);
    exchange_case("out_already_in", {0, 100}, 0, 1);
    exchange_case("in_out_of_range", {0, 100}, 5, 1);
    exchange_case("out_of_range", {0, 100}, 0, 5);

    // ---- cache_complement_blob_or_fallback ----------------------------------
    fallback_case("in_complement", {0, 100, 200}, 1, true);
    fallback_case("sentinel_goes_mapped", {0, kNoCacheComplement}, 1, true);
    fallback_case("no_complement_backing", {0, 100}, 1, false);
    fallback_case("index_past_end", {0, 100}, 5, true);

    // ---- choose_resident_keep_from ------------------------------------------
    keep_case("all_fit", {100, 100, 100}, 0, 1000, 0);
    keep_case("base_over_budget", {100}, 2000, 1000, 0);
    keep_case("partial", {100, 100, 100}, 0, 250, 0);
    keep_case("exact_fit", {100, 100}, 0, 200, 0);
    keep_case("lend_from_clamps", {100, 100, 100}, 0, 1000, 99);
    keep_case("lend_from_negative", {100, 100, 100}, 0, 1000, -1);
    keep_case("nothing_kept", {100, 100}, 0, 1000, 2);
    keep_case("zero_budget", {100}, 0, 0, 0);

    // ---- host_available_memory: the fake /proc + cgroup tree -----------------
    // The cgroup file's group path is absolute, and `root / abs` replaces the root,
    // so a group only resolves when its path is literally under the root - which is
    // what a real /sys/fs/cgroup looks like. The fixture paths are absolute too.
    host_case("no_cgroup_line", {{"cgroup_file", "MemAvailable: 16384 kB\n"}, {"groups", ""}});
    host_case("no_memavailable", {{"cgroup_file", "MemTotal: 32768 kB\n"}, {"groups", "0::/\n"}});
    host_case("groups_file_missing", {{"cgroup_file", "MemAvailable: 16384 kB\n"}});
    host_case("v2_root_no_controllers", {{"cgroup_file", "MemAvailable: 16384 kB\n"}, {"groups", "0::/\n"}});
    host_case("v2_max", {{"cgroup_file", "MemAvailable: 16384 kB\n"}, {"groups", "0::/foo\n"},
                         {"cgroup.controllers", "memory\n"}, {"foo/memory.max", "max\n"},
                         {"foo/memory.current", "1000\n"},
                         {"foo/memory.stat", "inactive_file 100\nfile_dirty 10\nfile_writeback 5\n"}});
    host_case("v2_limit", {{"cgroup_file", "MemAvailable: 16384 kB\n"}, {"groups", "0::/foo\n"},
                           {"cgroup.controllers", "memory\n"}, {"foo/memory.max", "8000\n"},
                           {"foo/memory.current", "1000\n"},
                           {"foo/memory.stat", "inactive_file 100\nfile_dirty 10\nfile_writeback 5\n"}});
    host_case("v2_tighter_than_meminfo", {{"cgroup_file", "MemAvailable: 16384 kB\n"}, {"groups", "0::/foo\n"},
                                          {"cgroup.controllers", "memory\n"}, {"foo/memory.max", "2000\n"},
                                          {"foo/memory.current", "1000\n"},
                                          {"foo/memory.stat", "inactive_file 100\nfile_dirty 10\nfile_writeback 5\n"}});
    host_case("v2_unreadable_limit", {{"cgroup_file", "MemAvailable: 16384 kB\n"}, {"groups", "0::/foo\n"},
                                      {"cgroup.controllers", "memory\n"}, {"foo/memory.current", "1000\n"},
                                      {"foo/memory.stat", "inactive_file 100\nfile_dirty 10\nfile_writeback 5\n"}});
    host_case("v2_missing_stat", {{"cgroup_file", "MemAvailable: 16384 kB\n"}, {"groups", "0::/foo\n"},
                                  {"cgroup.controllers", "memory\n"}, {"foo/memory.max", "8000\n"},
                                  {"foo/memory.current", "1000\n"}});
    host_case("v2_duplicate_key", {{"cgroup_file", "MemAvailable: 16384 kB\n"}, {"groups", "0::/foo\n"},
                                   {"cgroup.controllers", "memory\n"}, {"foo/memory.max", "8000\n"},
                                   {"foo/memory.current", "1000\n"},
                                   {"foo/memory.stat", "inactive_file 100\ninactive_file 200\nfile_dirty 10\nfile_writeback 5\n"}});
    host_case("v2_ancestor_and_child", {{"cgroup_file", "MemAvailable: 16384 kB\n"}, {"groups", "0::/foo/bar\n"},
                                        {"cgroup.controllers", "memory\n"}, {"foo/memory.max", "9000\n"},
                                        {"foo/memory.current", "1000\n"},
                                        {"foo/memory.stat", "inactive_file 100\nfile_dirty 10\nfile_writeback 5\n"},
                                        {"foo/bar/memory.max", "7000\n"}, {"foo/bar/memory.current", "1000\n"},
                                        {"foo/bar/memory.stat", "inactive_file 100\nfile_dirty 10\nfile_writeback 5\n"}});
    host_case("v1_memory_controller", {{"cgroup_file", "MemAvailable: 16384 kB\n"}, {"groups", "4:memory:/bar\n"},
                                       {"memory/bar/memory.limit_in_bytes", "6000\n"},
                                       {"memory/bar/memory.usage_in_bytes", "1000\n"}});
    host_case("v1_unlimited", {{"cgroup_file", "MemAvailable: 16384 kB\n"}, {"groups", "4:memory:/bar\n"},
                               {"memory/bar/memory.limit_in_bytes", "9223372036854771712\n"},
                               {"memory/bar/memory.usage_in_bytes", "1000\n"}});

    // the fixture tree last, so the paths are the ones the cases used
    std::printf("\n");
    for (const auto& kv : g_files) {
        std::printf("CORPUS|FILE|%s|", kv.first.c_str());
        for (unsigned char c : kv.second) std::printf("%02x", c);
        std::printf("\n");
    }
    return 0;
}
