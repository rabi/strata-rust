// Golden-trace generator for the native weight path: `src/core/weights.cpp`
// (the pack loader), `src/core/native_dense.cpp` (native GGUF projections) and
// the byte-format tables of `native_mmvq.hpp`.
//
// It runs the REAL code and prints every intermediate the Rust port claims to
// reproduce: index parse verdicts, per-plane segment plans, fnv1a64 of the exact
// bytes the loader hands to cudaMemcpy, fnv1a64 of the arena after every load,
// error strings, served_names sets, and upload plans.
//
// Build (the CUDA calls are --wrap'ped, so no device and no CUDA toolkit):
//   g++ -std=c++20 -w -I$S/include -I/tmp/cudastub -Itools -c tools/native_dense_corpus.cpp -o ndc.o
//   g++ -std=c++20 -w -I$S/include -c $S/src/core/weights.cpp -o w.o
//   g++ -std=c++20 -w -I$S/include -c $S/src/core/native_dense.cpp -o ndn.o
//   g++ -o ndcorpus ndc.o w.o ndn.o stubs.o native_mm_tables.o \
//       -Wl,--wrap=cudaMemcpy -Wl,--wrap=cudaMalloc -Wl,--wrap=cudaFree \
//       -Wl,--wrap=cudaHostAlloc -Wl,--wrap=cudaFreeHost \
//       -Wl,--wrap=cudaDeviceSynchronize
//   (native_mm_tables.cpp transcribes the three inline functions of
//    include/strata/kernels/native_mmvq.hpp — the .cu itself needs nvcc.)
//
// Output: one `CORPUS|<kind>|k=v|...` line per observation, byte-identical
// across runs and machines; the Rust test replays it against its own port.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "gguf_fixture.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <map>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::NativeDense;
using strata::core::WeightTable;
using strata::kernels::native_mmvq_supported;
using strata::kernels::native_mmvq_weight_bytes;
using strata::kernels::native_q8_1_bytes;

// ------------------------------------------------------------------ fake device
//
// cudaMalloc slices one host slab, cudaMemcpy is a memcpy, and every upload is
// recorded (source bytes fingerprinted, destination located) so the replay can
// compare the plan byte for byte. cudaHostAlloc/cudaFreeHost back the loader's
// pinned staging buffers with plain malloc/free.
namespace {

std::vector<uint8_t>& slab() {
    static std::vector<uint8_t> s(96ull << 20, 0);
    return s;
}
size_t slab_used = 0;
int malloc_calls = 0, memcpy_calls = 0;
int fail_malloc_at = 0, fail_memcpy_at = 0;

struct Write {
    std::string dst;
    uint64_t bytes, fnv_src;
};
std::vector<Write> writes;
bool recording = false;

struct Region {
    const uint8_t* base;
    size_t len;
    std::string name;
};
std::vector<Region> regions;
void reg(const void* p, size_t n, std::string name) {
    if (!p || !n) return;
    regions.push_back({static_cast<const uint8_t*>(p), n, std::move(name)});
}
std::string where(const void* p, size_t n) {
    if (!p) return "null";
    auto* b = static_cast<const uint8_t*>(p);
    for (const auto& r : regions)
        if (b >= r.base && b < r.base + r.len && b + n <= r.base + r.len)
            return r.name + ":" + std::to_string(b - r.base);
    return "outside";
}

constexpr uint64_t NE0 = 256, NE1 = 4;
const char* KEY = "blk.1.ple_key.weight";
const char* QKV = "blk.0.attn_qkv.weight";

std::vector<fixture::Kv> arch_keys() {
    return {fixture::str("general.architecture", "qwen4exp"),
            fixture::u32("qwen4exp.block_count", 48),
            fixture::u32("qwen4exp.embedding_length", 2560),
            fixture::u32("qwen4exp.expert_count", 512),
            fixture::u32("qwen4exp.expert_used_count", 10),
            fixture::u32("qwen4exp.attention.head_count", 24),
            fixture::u32("qwen4exp.attention.head_count_kv", 2)};
}

/// The corpus writes its fixtures under a fresh temp dir; the replay needs the
/// same bytes, so every file is walked at the end and printed as hex. `keep`
/// leaves the tree in place for a human to inspect (no path is printed unless a
/// file's content actually has to be dumped, so the golden stays stable).
struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-nd-corpus-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

/// Print every fixture file as `CORPUS|FILE|<rel>|<hex>`, ordered by relative
/// path so the golden is reproducible.
void dump_fixtures(TempDir& tmp) {
    std::vector<fs::path> files;
    for (const auto& e : fs::recursive_directory_iterator(tmp.path))
        if (e.is_regular_file()) files.push_back(e.path());
    std::sort(files.begin(), files.end());
    // Most of these files are the same 256- or 4096-byte pattern written by
    // every pack helper, so the hex goes out once under an id and the rest of
    // the lines reference it.
    std::map<std::string, int> blobs;
    for (const auto& p : files) {
        std::ifstream in(p, std::ios::binary);
        if (!in) continue;
        std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        std::error_code ec;
        const auto rel = fs::relative(p, tmp.path, ec).string();
        const std::string hex = fixture::hex(bytes.data(), bytes.size());
        auto it = blobs.find(hex);
        if (it == blobs.end()) {
            const int id = (int) blobs.size();
            blobs.emplace(hex, id);
            std::printf("CORPUS|BLOB|%d|%s\n", id, hex.c_str());
            std::printf("CORPUS|FILE|%s|%d\n", rel.c_str(), id);
        } else {
            std::printf("CORPUS|FILE|%s|%d\n", rel.c_str(), it->second);
        }
    }
}

// tests/core/native_dense_ple_key_test.cpp's fixture: a shape-only IQ3_XXS row for
// the qkv, and the key either as a raw BF16 row in dense.bin (index kind 4, the
// --compat-bf16 product) or as another shape-only row.
void write_pack(const fs::path& dir, bool bf16_key) {
    fs::create_directories(dir);
    const uint64_t key_bytes = NE0 * NE1 * 2;
    std::string idx = "# align 256 pool " + std::to_string(bf16_key ? 2048 + 256 : 256) + " tensors 2\n";
    idx += std::string(QKV) + " 0 0 0 0 0 0 256 4 8 0 32 0 0 0 0 0 0 0\n";
    if (bf16_key)
        idx += std::string(KEY) + " 0 4 0 " + std::to_string(key_bytes) + " 256 " + std::to_string(key_bytes) +
               " 256 4 0 0 1 0 0 0 0 0 0 0\n";
    else
        idx += std::string(KEY) + " 0 0 0 0 0 256 0 256 4 8 0 32 0 0 0 0 0 0 0\n";
    std::ofstream(dir / "index.txt", std::ios::binary) << idx;
    std::vector<char> bytes(bf16_key ? key_bytes : 16);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (char) fixture::pattern(9, i);
    std::ofstream(dir / "dense.bin", std::ios::binary).write(bytes.data(), (std::streamsize) bytes.size());
}

std::string join(const std::set<std::string>& s) {
    std::string out;
    for (const auto& n : s) out += n + ",";
    return out;
}

// The rows `native_dense` needs beyond the header: native_* and the skipped
// canonical bytes, which `WeightRef` keeps for exactly that check.
void print_ref(const char* tag, const std::string& name, const strata::core::WeightRef* w) {
    if (!w) {
        std::printf("CORPUS|REF|%s|%s|absent\n", tag, name.c_str());
        return;
    }
    std::printf("CORPUS|REF|%s|%s|bytes=%llu|ne=%lld,%lld|elements=%lld|kind=%d|quantized=%d"
                "|resident=%d|native=%d|native_type=%d|native_q8_1=%d|codes=%llu|scales=%llu"
                "|offset=%llu|fp16=%d|act=%d|src=%d:%llu+%llu|code_bits=%d|group=%d|bias=%d|iq4=%d\n",
                tag, name.c_str(), (unsigned long long) w->bytes, (long long) w->ne0, (long long) w->ne1,
                (long long) w->elements, (int) w->kind, w->quantized() ? 1 : 0, w->resident ? 1 : 0,
                w->native_data ? 1 : 0, w->native_type, w->native_q8_1 ? 1 : 0,
                (unsigned long long) w->codes_bytes, (unsigned long long) w->scales_bytes,
                (unsigned long long) w->offset_bytes, w->scales_fp16 ? 1 : 0, w->act_kind, w->file_id,
                (unsigned long long) w->src_off, (unsigned long long) w->src_bytes, w->code_bits,
                w->group_elems, w->code_bias, w->codebook_iq4nl ? 1 : 0);
}

/// A pack with one 256-byte row per name, loaded the way `generate.cpp` does it:
/// the native names are in the skip set, so the canonical bytes stay un-uploaded
/// and the row keeps its shape/metadata for `native_dense` to attach to.
void build_table(const fs::path& pack, const std::vector<std::string>& names,
                 const std::set<std::string>& skip, WeightTable& wt, void** arena,
                 std::string& err, uint64_t pool = 65536) {
    fs::create_directories(pack);
    std::string idx = "# align 256 pool " + std::to_string(pool) + " tensors " +
                      std::to_string(names.size()) + "\n";
    uint64_t at = 0;
    for (const auto& n : names) {
        idx += n + " 0 0 " + std::to_string(at) + " 256 " + std::to_string(at) +
               " 256 256 4 8 0 32 0 0 0 256 0 0 0 0 0 0\n";
        at += 256;
    }
    std::ofstream(pack / "index.txt", std::ios::binary) << idx;
    std::vector<char> bytes(256 * names.size());
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (char) fixture::pattern(11, i);
    std::ofstream(pack / "dense.bin", std::ios::binary).write(bytes.data(), (std::streamsize) bytes.size());
    if (cudaMalloc(arena, pool) != cudaSuccess) {
        *arena = nullptr;
        err = "cudaMalloc";
        return;
    }
    reg(*arena, pool, "arena:" + pack.filename().string());
    wt.load(pack.string(), *arena, pool, err, skip.empty() ? nullptr : &skip);
}

void print_writes(const char* tag) {
    for (const auto& w : writes)
        std::printf("CORPUS|WRITE|%s|%s|bytes=%llu|src_fnv=%llu\n", tag, w.dst.c_str(),
                    (unsigned long long) w.bytes, (unsigned long long) w.fnv_src);
}

extern "C" void native_block_size(int ggml_type, int* block_elems, int* block_bytes);
extern "C" int sizeof_block_q8_1();

// ------------------------------------------------------------------ scenario 1:
// byte-format tables, every supported type and every refusal message.
void scenario_bytes() {
    std::printf("CORPUS|Q81SIZE|bytes=%d\n", sizeof_block_q8_1());
    for (int t : {2, 6, 7, 8, 20, 42, 11, 12, 13, 14, 23, 16, 17, 18, 21, 22, 29, 0, 30, 255}) {
        int elems = 0, bytes = 0;
        native_block_size(t, &elems, &bytes);
        std::printf("CORPUS|BLOCKSIZE|t=%d|elems=%d|bytes=%d\n", t, elems, bytes);
    }
    const int types[] = {2, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 42};
    for (int t : types) {
        for (int n_in : {32, 64, 256, 512, 1024, 2560}) {
            std::string line = "CORPUS|BYTES|t=" + std::to_string(t) + "|n_in=" + std::to_string(n_in) +
                               "|supported=" + (native_mmvq_supported(t) ? "1" : "0");
            try {
                line += "|bytes=" + std::to_string(native_mmvq_weight_bytes(t, n_in, 4));
            } catch (const std::exception& e) {
                line += std::string("|throw=") + e.what();
            }
            std::printf("%s\n", line.c_str());
        }
    }
    const int bad[] = {0, 1, 3, 4, 5, 9, 10, 15, 19, 24, 25, 26, 27, 28, 30, 31, 41, 43, -1, 1000};
    for (int t : bad) {
        std::string line = "CORPUS|BYTES|t=" + std::to_string(t) + "|n_in=256|supported=0";
        try {
            line += "|bytes=" + std::to_string(native_mmvq_weight_bytes(t, 256, 4));
        } catch (const std::exception& e) {
            line += std::string("|throw=") + e.what();
        }
        std::printf("%s\n", line.c_str());
    }
    struct Bad {
        int t, n_in, n_out;
    };
    for (auto b : std::vector<Bad>{{2, 250, 4}, {42, 128, 0}, {8, 0, 1}, {14, 2147483621, 2147483647},
                                   {2, -32, 4}, {42, 64, -1}}) {
        std::string line = "CORPUS|BYTES|t=" + std::to_string(b.t) + "|n_in=" + std::to_string(b.n_in) +
                           "|n_out=" + std::to_string(b.n_out) + "|supported=" +
                           (native_mmvq_supported(b.t) ? "1" : "0");
        try {
            line += "|bytes=" + std::to_string(native_mmvq_weight_bytes(b.t, b.n_in, b.n_out));
        } catch (const std::exception& e) {
            line += std::string("|throw=") + e.what();
        }
        std::printf("%s\n", line.c_str());
    }
    for (auto p : std::vector<std::pair<int, int>>{{256, 1}, {2560, 8}, {32, 1}, {255, 1}, {256, 0}, {256, 9},
                                                    {2147483647, 2147483647}}) {
        std::string line = "CORPUS|Q81|n_in=" + std::to_string(p.first) + "|ncols=" + std::to_string(p.second);
        try {
            line += "|bytes=" + std::to_string(native_q8_1_bytes(p.first, p.second));
        } catch (const std::exception& e) {
            line += std::string("|throw=") + e.what();
        }
        std::printf("%s\n", line.c_str());
    }
}

// ------------------------------------------------------------------ scenario 2:
// the C++ test's fixture, in generate.cpp's order: served_names ->
// keep_unquantized_ple_key -> pool_bytes -> WeightTable::load -> NativeDense::load.
void scenario_fixture(TempDir& tmp, const char* tag, bool bf16_key, uint32_t key_type) {
    const fs::path pack = tmp.path / tag;
    write_pack(pack, bf16_key);
    const std::string shard = (tmp.path / (std::string(tag) + ".gguf")).string();
    fixture::write(shard, arch_keys(), {{QKV, {NE0, NE1}, 8, 1}, {KEY, {NE0, NE1}, key_type, 2}});

    std::set<std::string> skip;
    std::string err;
    bool ok = NativeDense::served_names({shard}, true, skip, err);
    std::printf("CORPUS|SERVED|%s|ok=%d|err=%s|names=%s\n", tag, ok ? 1 : 0, err.c_str(), join(skip).c_str());

    ok = NativeDense::keep_unquantized_ple_key(pack.string(), skip, err);
    std::printf("CORPUS|KEEP|%s|ok=%d|err=%s|key_skipped=%d|qkv_skipped=%d\n", tag, ok ? 1 : 0, err.c_str(),
                skip.count(KEY) ? 1 : 0, skip.count(QKV) ? 1 : 0);

    uint64_t pool = 0;
    ok = WeightTable::pool_bytes(pack.string(), pool, err, &skip);
    std::printf("CORPUS|POOL|%s|ok=%d|err=%s|pool=%llu\n", tag, ok ? 1 : 0, err.c_str(),
                (unsigned long long) pool);

    const uint64_t cap = pool ? pool : 256;
    void* arena = nullptr;
    if (cudaMalloc(&arena, cap) != cudaSuccess) {
        std::printf("CORPUS|FAIL|%s|cudaMalloc\n", tag);
        return;
    }
    reg(arena, cap, std::string("arena:") + tag);
    WeightTable wt;
    writes.clear();
    recording = true;
    ok = wt.load(pack.string(), arena, cap, err, &skip);
    recording = false;
    std::printf("CORPUS|LOAD|%s|ok=%d|err=%s|writes=%zu\n", tag, ok ? 1 : 0, err.c_str(), writes.size());
    print_writes(tag);
    std::printf("CORPUS|REPORT|%s|tensors=%zu|arena=%llu|re_rounded=%zu|bytes_saved=%llu\n", tag,
                wt.report().tensors, (unsigned long long) wt.report().arena_bytes, wt.report().re_rounded,
                (unsigned long long) wt.report().bytes_saved);
    print_ref(tag, QKV, wt.find(QKV));
    print_ref(tag, KEY, wt.find(KEY));
    std::printf("CORPUS|ARENA|%s|fnv=%llu\n", tag, (unsigned long long) strata::core::fnv1a64((uint8_t*) arena, cap));

    NativeDense dense;
    writes.clear();
    recording = true;
    ok = dense.load({shard}, wt, err, true);
    recording = false;
    uint64_t upload_bytes = 0;
    for (auto& w : writes) upload_bytes += w.bytes;
    std::printf("CORPUS|DENSE|%s|ok=%d|err=%s|tensors=%zu|weight_bytes=%llu|uploads=%zu|upload_bytes=%llu\n",
                tag, ok ? 1 : 0, err.c_str(), dense.tensor_count(),
                (unsigned long long) dense.weight_bytes(), writes.size(),
                (unsigned long long) upload_bytes);
    print_writes(tag);
    print_ref(tag, QKV, wt.find(QKV));
    print_ref(tag, KEY, wt.find(KEY));
    std::printf("CORPUS|ARENA|%s|fnv=%llu\n", tag, (unsigned long long) strata::core::fnv1a64((uint8_t*) arena, cap));
    cudaFree(arena);
}

// ------------------------------------------------------------------ scenario 3:
// plan v0.3 P6's tensor kinds through the real loader — a quantized plane split,
// an fp16 scale plane widened, a BF16 promotion shrunk, an f16 promotion rounded
// (a real f32->f16 conversion, denormal and NaN regimes included), raw16 kinds
// 4/5 copied — plus every refusal the loader can print.
void scenario_kinds(TempDir& tmp) {
    const fs::path pack = tmp.path / "kinds";
    fs::create_directories(pack);

    // columns: name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias
    //          group iq4 has_offset scales_fp16 act_kind
    auto row = [](const char* name, int file, int kind, uint64_t src_off, uint64_t src_bytes,
                  uint64_t dst_off, uint64_t dst_bytes, int64_t ne0, int64_t ne1, int code_bits,
                  uint64_t codes, uint64_t scales, uint64_t offset, int fp16) {
        return std::string(name) + " " + std::to_string(file) + " " + std::to_string(kind) + " " +
               std::to_string(src_off) + " " + std::to_string(src_bytes) + " " + std::to_string(dst_off) +
               " " + std::to_string(dst_bytes) + " " + std::to_string(ne0) + " " + std::to_string(ne1) +
               " " + std::to_string(code_bits) + " 0 32 0 " + std::to_string(offset ? 1 : 0) + " " +
               std::to_string(codes) + " " + std::to_string(scales) + " " + std::to_string(offset) + " " +
               std::to_string(fp16) + " 0 0 0 0 0 0\n";
    };

    // one 128-byte window per row's source span, so every byte a segment reads is
    // predictable; f32 windows hold values that exercise rounding both ways.
    std::vector<char> file(64 * 1024, 0);
    for (size_t i = 0; i < file.size(); ++i) file[i] = (char) fixture::pattern(3, i);
    // an f32 window for the promotion rows: values spanning normal, exactly-tied,
    // underflow and NaN so the f32->f16 path is compared on all four
    auto put_f32 = [&](size_t at, std::vector<float> vs) {
        for (size_t i = 0; i < vs.size(); ++i) std::memcpy(&file[at + i * 4], &vs[i], 4);
    };
    put_f32(128, {1.0f, -2.5f, 65504.0f, 65520.0f, 1.0f / 131072, 1.0f / 16777216,
                  std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                  1.0000076f, 0.0f, -0.0f, 3.0f});
    put_f32(640, {1.0f, 2.0f, 1.5f, 0.25f, 1.0f / 3, 7.0f, -7.0f, 123.456f});
    std::ofstream(pack / "dense.bin", std::ios::binary).write(file.data(), (std::streamsize) file.size());
    std::ofstream(pack / "extra.bin", std::ios::binary).write(file.data(), (std::streamsize) file.size());

    std::string idx = "# align 64 pool 65536 tensors 8\n";
    // quantized, 3 planes, fp16 scales in the pack: codes + widened scales + offset
    idx += row("t.q", 0, 0, 0, 120, 0, 128, 128, 1, 8, 96, 16, 16, 1);
    // kind 1 (bf16 promoted to f32) -> high 16 bits
    idx += row("t.bf", 0, 1, 128, 48, 128, 24, 12, 1, 0, 0, 0, 0, 0);
    // kind 3 (f16 promoted to f32) -> a real conversion
    idx += row("t.h", 0, 3, 128, 48, 152, 24, 12, 1, 0, 0, 0, 0, 0);
    // kind 2 (f32 verbatim)
    idx += row("t.f32", 0, 2, 640, 32, 176, 32, 8, 1, 0, 0, 0, 0, 0);
    // kind 4/5: already 16 bits in the file, copied straight through
    idx += row("t.raw4", 0, 4, 768, 64, 208, 64, 32, 1, 0, 0, 0, 0, 0);
    idx += row("t.raw5", 3, 5, 832, 64, 272, 64, 64, 1, 0, 0, 0, 0, 0);
    // a quantized row with no offset plane and no scales at all (codes only)
    idx += row("t.codes", 0, 0, 896, 64, 336, 64, 256, 1, 2, 64, 0, 0, 0);
    std::ofstream(pack / "index.txt", std::ios::binary) << idx;

    void* arena = nullptr;
    if (cudaMalloc(&arena, 65536) != cudaSuccess) {
        std::printf("CORPUS|FAIL|kinds|cudaMalloc\n");
        return;
    }
    reg(arena, 65536, "arena:kinds");
    WeightTable wt;
    std::string err;
    writes.clear();
    recording = true;
    bool ok = wt.load(pack.string(), arena, 65536, err);
    recording = false;
    std::printf("CORPUS|LOAD|kinds|ok=%d|err=%s|writes=%zu\n", ok ? 1 : 0, err.c_str(), writes.size());
    print_writes("kinds");
    std::printf("CORPUS|REPORT|kinds|tensors=%zu|arena=%llu|re_rounded=%zu|bytes_saved=%llu\n",
                wt.report().tensors, (unsigned long long) wt.report().arena_bytes, wt.report().re_rounded,
                (unsigned long long) wt.report().bytes_saved);
    for (const char* n : {"t.q", "t.bf", "t.h", "t.f32", "t.raw4", "t.raw5", "t.codes"})
        print_ref("kinds", n, wt.find(n));
    auto* a = static_cast<uint8_t*>(arena);
    std::printf("CORPUS|ARENA|kinds|q=%llu|bf=%llu|h=%llu|f32=%llu|r4=%llu|r5=%llu|codes=%llu\n",
                (unsigned long long) strata::core::fnv1a64(a + 0, 128),
                (unsigned long long) strata::core::fnv1a64(a + 128, 24),
                (unsigned long long) strata::core::fnv1a64(a + 152, 24),
                (unsigned long long) strata::core::fnv1a64(a + 176, 32),
                (unsigned long long) strata::core::fnv1a64(a + 208, 64),
                (unsigned long long) strata::core::fnv1a64(a + 272, 64),
                (unsigned long long) strata::core::fnv1a64(a + 336, 64));
    // the promoted rows' engine bytes verbatim: the f32->f16 port is judged here
    std::printf("CORPUS|HEX|kinds|bf=%s|h=%s\n",
                fixture::hex(a + 128, 24).c_str(), fixture::hex(a + 152, 24).c_str());
    cudaFree(arena);

    // one bad index per case: the exact message the loader prints
    const std::string head = "# align 64 pool 65536 tensors 1\n";
    const std::vector<std::pair<std::string, std::string>> bads = {
        {"fields", head + "short 0 0 0 0 0\n"},
        {"header_fields", "# align 64 pool 7 tensors 1\n"},
        {"shape_only", head + row("t.s", 0, 0, 0, 128, 0, 0, 32, 1, 8, 96, 16, 16, 1)},
        {"no_header", row("t.q", 0, 0, 0, 128, 0, 128, 128, 1, 8, 96, 16, 16, 1)},
        {"odd_fp16_plane", head + row("t.o", 0, 0, 0, 121, 0, 128, 128, 1, 8, 97, 15, 16, 1)},
        {"promote_mismatch", head + row("t.p", 0, 1, 128, 44, 128, 24, 12, 1, 0, 0, 0, 0, 0)},
        {"seg_mismatch", head + row("t.g", 0, 0, 0, 128, 0, 128, 128, 1, 8, 96, 16, 0, 0)},
        {"bad_file", head + row("t.x", 9, 0, 0, 128, 0, 128, 128, 1, 8, 96, 16, 16, 1)},
        {"good", head + row("t.q", 0, 0, 0, 120, 0, 128, 128, 1, 8, 96, 16, 16, 1)},
    };
    for (const auto& b : bads) {
        const fs::path bp = tmp.path / ("bad-" + b.first);
        fs::create_directories(bp);
        std::ofstream(bp / "index.txt", std::ios::binary) << b.second;
        std::ofstream(bp / "dense.bin", std::ios::binary).write(file.data(), 4096);
        std::ofstream(bp / "extra.bin", std::ios::binary).write(file.data(), 4096);
        uint64_t pool = 0;
        std::string perr;
        const bool pok = WeightTable::pool_bytes(bp.string(), pool, perr);
        std::printf("CORPUS|POOL|bad-%s|ok=%d|err=%s|pool=%llu\n", b.first.c_str(), pok ? 1 : 0,
                    perr.c_str(), (unsigned long long) pool);
        // the arena-too-small branch runs against the "good" index only
        const uint64_t cap = b.first == "arena_small" ? 4 : 65536;
        void* ar = nullptr;
        cudaMalloc(&ar, 65536);
        reg(ar, 65536, "arena:bad-" + b.first);
        WeightTable wt2;
        writes.clear();
        recording = true;
        std::string lerr;
        const bool lok = wt2.load(bp.string(), ar, cap, lerr);
        recording = false;
        std::printf("CORPUS|LOAD|bad-%s|ok=%d|err=%s|writes=%zu|tensors=%zu\n", b.first.c_str(),
                    lok ? 1 : 0, lerr.c_str(), writes.size(), wt2.report().tensors);
        cudaFree(ar);
        int bits = 99;
        std::string ierr;
        const bool iok = WeightTable::index_code_bits(bp.string(), "t.q", bits, ierr);
        std::printf("CORPUS|BITS|bad-%s|ok=%d|err=%s|bits=%d\n", b.first.c_str(), iok ? 1 : 0,
                    ierr.c_str(), bits);
    }

    // the arena-too-small message, with an otherwise good index
    {
        const fs::path bp = tmp.path / "bad-arena_small";
        fs::create_directories(bp);
        std::ofstream(bp / "index.txt", std::ios::binary) << head + row("t.q", 0, 0, 0, 120, 0, 128, 128, 1, 8, 96, 16, 16, 1);
        std::ofstream(bp / "dense.bin", std::ios::binary).write(file.data(), 4096);
        void* ar = nullptr;
        cudaMalloc(&ar, 65536);
        reg(ar, 65536, "arena:arena_small");
        WeightTable wt3;
        writes.clear();
        recording = true;
        std::string lerr;
        const bool lok = wt3.load(bp.string(), ar, 4, lerr);
        recording = false;
        std::printf("CORPUS|LOAD|bad-arena_small|ok=%d|err=%s|writes=%zu\n", lok ? 1 : 0, lerr.c_str(),
                    writes.size());
        cudaFree(ar);
    }

    // index_code_bits on a missing directory, and a missing tensor row
    int bits = 99;
    std::string ierr;
    bool iok = WeightTable::index_code_bits((tmp.path / "nowhere").string(), KEY, bits, ierr);
    std::printf("CORPUS|BITS|no_index|ok=%d|err=%s|bits=%d\n", iok ? 1 : 0, ierr.c_str(), bits);
    iok = WeightTable::index_code_bits(pack.string(), "t.absent", bits, ierr);
    std::printf("CORPUS|BITS|no_row|ok=%d|err=%s|bits=%d\n", iok ? 1 : 0, ierr.c_str(), bits);
    // keep_unquantized_ple_key when the key is not in skip (the index is never opened)
    std::set<std::string> skip_nokey{QKV};
    std::string kerr;
    NativeDense::keep_unquantized_ple_key((tmp.path / "nowhere").string(), skip_nokey, kerr);
    std::printf("CORPUS|KEEP|no_key_in_skip|err=%s|skipped=%zu\n", kerr.c_str(), skip_nokey.size());
}

// ------------------------------------------------------------------ scenario 4:
// the span checks. `write_raw` records offsets and a payload size the loader
// cannot reconcile, which is the only way to reach those five refusals.
void scenario_spans(TempDir& tmp) {
    struct Span {
        const char* tag;
        std::vector<fixture::Raw> tensors;
        uint64_t payload;
    };
    // Q8_0, [256,4] -> 1088 B; [256,1] -> 272 B
    const std::vector<Span> spans = {
        {"truncated", {{QKV, {NE0, NE1}, 8, 0}}, 512},
        // one tensor's bytes run past the next tensor's offset
        {"overlap", {{QKV, {NE0, NE1}, 8, 0}, {"blk.1.attn_q.weight", {NE0, 1}, 8, 256}}, 4096},
        // two directory entries naming the same offset
        {"same_offset", {{QKV, {NE0, 1}, 8, 0}, {"blk.1.attn_q.weight", {NE0, 1}, 8, 0}}, 4096},
        // a tensor with no dimensions at all
        {"no_dims", {{QKV, {}, 8, 0}}, 4096},
        // ne0 not a whole number of blocks
        {"bad_ne0", {{QKV, {250, 4}, 8, 0}}, 4096},
        // a type the block-geometry table does not know
        {"bad_type", {{QKV, {NE0, NE1}, 999, 0}}, 4096},
        // a zero dimension
        {"zero_dim", {{QKV, {NE0, 0}, 8, 0}}, 4096},
        // blocks * block_bytes overflows u64: shape[0] = 2^32, ne1 = 2^32 - 1
        {"byte_overflow", {{QKV, {1ull << 32, (1ull << 32) - 1}, 8, 0}}, 4096},
    };
    for (const auto& sp : spans) {
        const std::string shard = (tmp.path / (std::string("span_") + sp.tag + ".gguf")).string();
        fixture::write_raw(shard, arch_keys(), sp.tensors, sp.payload);
        std::set<std::string> skip;
        std::string serr;
        const bool sok = NativeDense::served_names({shard}, true, skip, serr);
        NativeDense dense;
        WeightTable wt;
        void* arena = nullptr;
        std::string lerr;
        build_table(tmp.path / (std::string("span_") + sp.tag), {QKV, "blk.1.attn_q.weight"}, skip, wt, &arena, lerr);
        std::string err;
        const bool ok = dense.load({shard}, wt, err, true);
        std::printf("CORPUS|SPAN|%s|served=%d|served_err=%s|ok=%d|err=%s\n", sp.tag, sok ? 1 : 0,
                    serr.c_str(), ok ? 1 : 0, err.c_str());
        if (arena) cudaFree(arena);
    }

    // an architecture key that is present and wrong: check_architecture's
    // message comes back verbatim, without the "native dense: " prefix
    const std::string wrong = (tmp.path / "arch_wrong.gguf").string();
    auto keys = arch_keys();
    for (auto& k : keys)
        if (k.key == "qwen4exp.block_count") k.nums[0] = 24;
    fixture::write(wrong, keys, {{QKV, {NE0, NE1}, 8, 1}});
    std::string err;
    NativeDense dense;
    WeightTable wt;
    void* arena = nullptr;
    std::string lerr;
    build_table(tmp.path / "arch_wrong", {QKV}, {}, wt, &arena, lerr);
    const bool ok = dense.load({wrong}, wt, err, true);
    std::printf("CORPUS|ARCH|wrong|ok=%d|err=%s\n", ok ? 1 : 0, err.c_str());
    if (arena) cudaFree(arena);

    // a shard with no general.architecture and no split keys at all
    const std::string naked = (tmp.path / "arch_naked.gguf").string();
    fixture::write(naked, {}, {{QKV, {NE0, NE1}, 8, 1}});
    NativeDense d2;
    const bool ok2 = d2.load({naked}, wt, err, true);
    std::printf("CORPUS|ARCH|naked|ok=%d|err=%s\n", ok2 ? 1 : 0, err.c_str());

    // an unsupported type that is still an eligible name: the `continue` after
    // the table lookup (F32 = 0), and a native name whose matrix is not 2-D
    const std::string unsup = (tmp.path / "unsupported.gguf").string();
    fixture::write(unsup, arch_keys(), {{QKV, {NE0, NE1}, 0, 1}});
    NativeDense d3;
    const bool ok3 = d3.load({unsup}, wt, err, true);
    std::printf("CORPUS|UNSUPPORTED|ok=%d|err=%s|tensors=%zu\n", ok3 ? 1 : 0, err.c_str(),
                d3.tensor_count());

    // the same shard loaded twice against one table: the second object sees a
    // reference that already carries native bytes
    const std::string dup = (tmp.path / "attached.gguf").string();
    fixture::write(dup, arch_keys(), {{QKV, {NE0, NE1}, 8, 1}});
    WeightTable wt2;
    void* arena2 = nullptr;
    build_table(tmp.path / "attached", {QKV}, {}, wt2, &arena2, lerr);
    NativeDense first_load;
    const bool one = first_load.load({dup}, wt2, err, true);
    NativeDense second_load;
    const bool two = second_load.load({dup}, wt2, err, true);
    std::printf("CORPUS|ATTACHED|first=%d|second=%d|err=%s\n", one ? 1 : 0, two ? 1 : 0, err.c_str());
    if (arena2) cudaFree(arena2);

    // the same eligible name in two shards, both architecture-valid
    const std::string a = (tmp.path / "dup_a.gguf").string();
    const std::string b = (tmp.path / "dup_b.gguf").string();
    fixture::write(a, arch_keys(), {{QKV, {NE0, NE1}, 8, 1}});
    fixture::write(b, arch_keys(), {{QKV, {NE0, NE1}, 8, 2}});
    NativeDense d4;
    const bool four = d4.load({a, b}, wt, err, true);
    std::printf("CORPUS|DUP_NAME|ok=%d|err=%s\n", four ? 1 : 0, err.c_str());
}

// ------------------------------------------------------------------ scenario 4:
// shard arbitration: split metadata, continuation shards, duplicates, a stage's
// layer range, and the already-loaded guard.
void scenario_shards(TempDir& tmp) {
    const std::string first = (tmp.path / "sh0.gguf").string();
    auto with_split = [&](uint64_t no, uint64_t count) {
        auto kv = arch_keys();
        auto sk = fixture::split_keys(no, count, 2);
        kv.insert(kv.end(), sk.begin(), sk.end());
        return kv;
    };
    fixture::write(first, with_split(0, 2), {{QKV, {NE0, NE1}, 8, 1}, {KEY, {NE0, NE1}, 8, 2}});
    // a continuation shard carrying no architecture key of its own
    const std::string cont = (tmp.path / "sh1.gguf").string();
    fixture::write(cont, fixture::split_keys(1, 2, 2),
                   {{"blk.1.attn_q.weight", {NE0, NE1}, 8, 3}});
    // a continuation whose split_tensors disagrees
    const std::string cont_bad = (tmp.path / "sh2.gguf").string();
    fixture::write(cont_bad, fixture::split_keys(1, 2, 3), {{"blk.1.attn_q.weight", {NE0, NE1}, 8, 4}});
    // a shard with a truncated / overlapping tensor, to exercise the span checks
    const std::string bad_span = (tmp.path / "span.gguf").string();
    fixture::write(bad_span, arch_keys(), {{QKV, {NE0, NE1}, 8, 5}}, 32);

    struct Case {
        std::vector<std::string> shards;
        bool ple;
        int64_t lo, hi;
        const char* why;
    };
    const std::vector<Case> cases = {
        {{first}, true, 0, -1, "single"},
        {{first, cont}, true, 0, -1, "continuation"},
        {{first, cont_bad}, true, 0, -1, "bad_continuation"},
        {{first, first}, true, 0, -1, "duplicate_number"},
        {{cont}, true, 0, -1, "no_architecture"},
        {{first}, true, 1, 2, "stage_holds_layer1"},
        {{first}, false, 0, -1, "ple_not_native"},
        {{}, true, 0, -1, "no_shard"},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        std::set<std::string> skip;
        std::string err;
        const bool sok = NativeDense::served_names(c.shards, c.ple, skip, err);
        std::printf("CORPUS|SERVED|c%zu_%s|ok=%d|err=%s|names=%s\n", i, c.why, sok ? 1 : 0, err.c_str(),
                    join(skip).c_str());

        const fs::path pack = tmp.path / ("c" + std::to_string(i));
        WeightTable wt;
        void* arena = nullptr;
        std::string lerr;
        build_table(pack, {QKV, KEY, "blk.1.attn_q.weight"}, skip, wt, &arena, lerr);
        const bool lok = arena && wt.report().tensors > 0;
        NativeDense dense;
        writes.clear();
        recording = true;
        const bool ok = dense.load(c.shards, wt, err, c.ple, c.lo, c.hi);
        recording = false;
        std::printf("CORPUS|DENSE|c%zu_%s|table_ok=%d|table_err=%s|ok=%d|err=%s|tensors=%zu"
                    "|weight_bytes=%llu|uploads=%zu\n",
                    i, c.why, lok ? 1 : 0, lerr.c_str(), ok ? 1 : 0, err.c_str(), dense.tensor_count(),
                    (unsigned long long) dense.weight_bytes(), writes.size());
        print_writes(c.why);
        if (arena) cudaFree(arena);
    }

    // the process-wide layer range, read by the NEXT load (the C++ keeps it in a
    // file-static; the Rust port takes it as an argument)
    NativeDense::set_layer_range(0, 1);
    {
        const std::string shard = (tmp.path / "range.gguf").string();
        fixture::write(shard, arch_keys(),
                       {{QKV, {NE0, NE1}, 8, 1}, {"blk.2.attn_q.weight", {NE0, NE1}, 8, 2}});
        std::set<std::string> skip;
        std::string serr;
        NativeDense::served_names({shard}, false, skip, serr);
        WeightTable wt;
        void* arena = nullptr;
        std::string terr;
        build_table(tmp.path / "range", {QKV, "blk.2.attn_q.weight"}, skip, wt, &arena, terr);
        NativeDense dense;
        writes.clear();
        recording = true;
        const bool ok = dense.load({shard}, wt, serr, false);
        recording = false;
        std::printf("CORPUS|RANGE|ok=%d|err=%s|tensors=%zu|uploads=%zu\n", ok ? 1 : 0, serr.c_str(),
                    dense.tensor_count(), writes.size());
        std::string err2;
        const bool again = dense.load({shard}, wt, err2, false);
        std::printf("CORPUS|AGAIN|ok=%d|err=%s\n", again ? 1 : 0, err2.c_str());
        if (arena) cudaFree(arena);
    }
    NativeDense::set_layer_range(-1, -1);

    // an eligible name that is not in the canonical table, and an unsupported
    // type: the two branches after the duplicate check
    {
        const std::string shard = (tmp.path / "absent.gguf").string();
        // an eligible name (`.attn_gate.weight`) the canonical table does not hold
        fixture::write(shard, arch_keys(),
                       {{"blk.0.attn_gate.weight", {NE0, NE1}, 8, 6}, {QKV, {NE0, NE1}, 8, 7}});
        WeightTable wt;
        void* arena = nullptr;
        std::string lerr;
        build_table(tmp.path / "absent", {QKV}, {}, wt, &arena, lerr);
        NativeDense dense;
        const bool ok = dense.load({shard}, wt, lerr, false);
        std::printf("CORPUS|ABSENT|ok=%d|err=%s\n", ok ? 1 : 0, lerr.c_str());
        // the same shard against a table that DOES hold the gate: it uploads
        WeightTable wt2;
        void* arena2 = nullptr;
        build_table(tmp.path / "absent2", {QKV, "blk.0.attn_gate.weight"}, {}, wt2, &arena2, lerr);
        NativeDense dense2;
        writes.clear();
        recording = true;
        const bool ok2 = dense2.load({shard}, wt2, lerr, false);
        recording = false;
        std::printf("CORPUS|PRESENT|ok=%d|err=%s|uploads=%zu\n", ok2 ? 1 : 0, lerr.c_str(), writes.size());
        print_writes("present");
        if (arena) cudaFree(arena);
        if (arena2) cudaFree(arena2);
    }
}

}  // namespace

int main() {
    TempDir tmp;
    std::printf("# native_dense corpus — %s\n", tmp.path.c_str());
    scenario_bytes();
    scenario_fixture(tmp, "orca", true, 18);  // --compat-bf16 key row + IQ3_XXS GGUF key
    scenario_fixture(tmp, "q8", false, 8);    // a quantized key row + Q8_0 GGUF key
    scenario_kinds(tmp);
    scenario_shards(tmp);
    scenario_spans(tmp);
    dump_fixtures(tmp);
    return 0;
}

// ------------------------------------------------------------------ the CUDA seam
extern "C" cudaError_t __wrap_cudaMalloc(void** p, size_t n) {
    ++malloc_calls;
    if (fail_malloc_at && malloc_calls == fail_malloc_at) return cudaErrorMemoryAllocation;
    auto& s = slab();
    if (slab_used + n > s.size()) return cudaErrorMemoryAllocation;
    *p = s.data() + slab_used;
    std::memset(*p, 0xa5, n);  // poison: an unread destination stays visible
    slab_used = (slab_used + n + 255) & ~(size_t) 255;
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaFree(void*) {
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaHostAlloc(void** p, size_t n, unsigned) {
    *p = std::malloc(n);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
extern "C" cudaError_t __wrap_cudaFreeHost(void* p) {
    std::free(p);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    return cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t) {
    return "stub";
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    ++memcpy_calls;
    if (fail_memcpy_at && memcpy_calls == fail_memcpy_at) return cudaErrorMemoryAllocation;
    if (recording)
        writes.push_back({where(dst, n), n, strata::core::fnv1a64(static_cast<const uint8_t*>(src), n)});
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
