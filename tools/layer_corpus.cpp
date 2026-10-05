// tools/layer_corpus.cpp — drive layer.cpp's deterministic surface and print what it says.
//
// layer.cpp is #included, not linked: `sform_of`, `plane_ptrs`, `Cursor`, `kv_plan` and `kv_pool_bytes` are in
// anonymous namespaces and would otherwise be unreachable.
//
// The four `strata::kernels` size helpers below are TRANSCRIBED from their .cu files (each cites the line)
// because this corpus builds without nvcc.  A misread of one of those is invisible here.  Everything else is
// the real code.
//
// Build (see the bottom of this file) and run; stdout is the golden.

#include "strata/core/layer.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::kernels {

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

// gr.cu:304
size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    const size_t sz[5] = {
        hc_dim * sizeof(float),
        hc_dim * sizeof(uint16_t),
        (size_t) s.hc_lr * sizeof(uint16_t),
        hc_dim * sizeof(float),
        (size_t) s.hc_lr * sizeof(float),
    };
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t) 15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*) base;
        void* ptr[5];
        for (int k = 0; k < 5; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float*) ptr[0];
        out.xq = (uint16_t*) ptr[1];
        out.lq = (uint16_t*) ptr[2];
        out.gated = (float*) ptr[3];
        out.lo = (float*) ptr[4];
    }
    return bytes;
}

}  // namespace strata::kernels

#include "layer.cpp"

using namespace strata;
using namespace strata::core;

static uint8_t* const BASE = (uint8_t*) (uintptr_t) 0x100000;
static std::string self;

static void off(const char* tag, const void* p) {
    std::printf("%s|%llu\n", tag, (unsigned long long) ((const uint8_t*) p - BASE));
}

static void show_layout(const char* section, const char* tag, uint64_t bytes_side, uint64_t init_total) {
    std::printf("LAYOUT|%s|%s|bytes=%llu|init=%llu\n", section, tag, (unsigned long long) bytes_side,
                (unsigned long long) init_total);
}

// ---------------------------------------------------------------- the two gates

static void gate_sform(const char* tag, const WeightRef& r) {
    kernels::SForm f;
    std::string err;
    bool ok = sform_of(r, f, "t", err);
    std::printf("SFORM|%s|ok=%d|bits=%d|bias=%d|group=%d|codebook=%d|has_off=%d|act=%d|err=%s\n", tag, ok,
                f.code_bits, f.code_bias, f.group_elems, (int) f.codebook, (int) f.has_offset, f.act_kind,
                err.c_str());
}

static void gate_planes(const char* tag, const WeightRef& r) {
    Planes p;
    std::string err;
    bool ok = plane_ptrs(r, "t", p, err);
    std::printf("PLANES|%s|ok=%d|codes=%llu|scales=%llu|offset=%lld|err=%s\n", tag, ok,
                (unsigned long long) ((const uint8_t*) p.codes - BASE),
                (unsigned long long) ((const uint8_t*) p.scales - BASE),
                p.offset ? (long long) ((const uint8_t*) p.offset - BASE) : -1LL, err.c_str());
}

// ---------------------------------------------------------------- the plans

static void plan(const char* tag, const kernels::QsaShapes& s, int64_t max_cells, int64_t ring,
                int64_t resident, bool ring_off, bool main_off) {
    g_kv_resident = resident;
    // the two env switches are read once per process through function statics, so each row needs its own
    // process: main re-execs itself with the env set and lets the child print its one row.
    if (ring_off) ::setenv("STRATA_KV_RING_OFF", "1", 1);
    else ::unsetenv("STRATA_KV_RING_OFF");
    if (main_off) ::setenv("STRATA_KV_MAIN_OFF", "1", 1);
    else ::unsetenv("STRATA_KV_MAIN_OFF");
    const KvPlan p = kv_plan(s, max_cells, ring);
    std::printf("KVPLAN|%s|mode=%d|pages=%lld|slots=%lld|pooled=%lld\n", tag, p.mode, (long long) p.pages,
                (long long) p.slots, (long long) p.pooled_rows);
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    (void) argc;
    self = argv[0];
    // the kv_plan rows come from re-executed children writing straight to the pipe; line-buffered parent
    // output would land after them.
    setvbuf(stdout, nullptr, _IONBF, 0);
    ModelGeometry g;
    kernels::QsaShapes s = qsa_shapes(g);

    // each kv_plan row needs a clean process for the env statics; the child prints its one row and leaves
    if (const char* only = ::getenv("CORPUS_PLAN_ONLY")) {
        plan(only, s, ::getenv("CORPUS_MAX") ? atoll(::getenv("CORPUS_MAX")) : 4096,
             ::getenv("CORPUS_RING") ? atoll(::getenv("CORPUS_RING")) : -1,
             ::getenv("CORPUS_RESIDENT") ? atoll(::getenv("CORPUS_RESIDENT")) : 0,
             ::getenv("CORPUS_RING_OFF") != nullptr, ::getenv("CORPUS_MAIN_OFF") != nullptr);
        return 0;
    }

    std::printf("Q8K|0|%llu\n", (unsigned long long) q8k_bytes(0));
    for (int64_t n : {256LL, 512LL, 255LL, 1024LL, 2560LL, 6144LL, 128LL})
        std::printf("Q8K|%lld|%llu\n", (long long) n, (unsigned long long) q8k_bytes(n));
    for (uint64_t n : {0ull, 1ull, 15ull, 16ull, 17ull, 4095ull})
        std::printf("ALIGN16|%llu|%llu\n", (unsigned long long) n, (unsigned long long) align_up16(n));

    // the gates: a quantized ref that adds up, and each way it can refuse
    WeightRef ok{};
    ok.code_bits = 4;
    ok.code_bias = -16;
    ok.group_elems = 32;
    ok.codebook_iq4nl = true;
    ok.has_offset = true;
    ok.act_kind = 1;
    ok.codes_bytes = 1024;
    ok.scales_bytes = 256;
    ok.offset_bytes = 128;
    ok.bytes = 1408;
    ok.kind = WeightKind::Verbatim;
    ok.data = BASE;
    gate_sform("iq4xs", ok);
    gate_planes("iq4xs", ok);

    WeightRef q2 = ok;
    q2.code_bits = 2;
    q2.codebook_iq4nl = false;
    gate_sform("q2_0", q2);
    gate_planes("q2_0", q2);

    WeightRef q8 = ok;
    q8.code_bits = 8;
    q8.has_offset = false;
    q8.offset_bytes = 0;
    q8.bytes = q8.codes_bytes + q8.scales_bytes;
    gate_sform("q8_0", q8);
    gate_planes("q8_0", q8);

    WeightRef notq = ok;
    notq.code_bits = 0;
    notq.kind = WeightKind::F32;
    gate_sform("f32", notq);

    WeightRef badbits = ok;
    badbits.code_bits = 3;
    gate_planes("bits3", badbits);

    WeightRef shorty = ok;
    shorty.bytes = 1407;
    gate_planes("sum_off_by_one", shorty);

    WeightRef nocodes = ok;
    nocodes.codes_bytes = 0;
    nocodes.bytes = nocodes.scales_bytes + nocodes.offset_bytes;
    gate_planes("zero_codes", nocodes);

    WeightRef liesaboutoffset = ok;
    liesaboutoffset.has_offset = false;
    gate_planes("has_offset_false", liesaboutoffset);

    WeightRef nooffsetbytes = ok;
    nooffsetbytes.has_offset = true;
    nooffsetbytes.offset_bytes = 0;
    nooffsetbytes.bytes = nooffsetbytes.codes_bytes + nooffsetbytes.scales_bytes;
    gate_planes("has_offset_true_zero", nooffsetbytes);

    // the arena layouts, real geometry and a small one where the parts are not 16-aligned
    ModelGeometry small;
    small.n_embd = 250;
    small.ssm_conv_channels = 1022;
    small.ssm_value_dim = 6142;
    small.n_head = 3;
    small.n_head_kv = 1;
    small.head_dim = 64;
    small.idx_q_heads = 2;
    small.idx_key_dim = 62;
    small.hc = 2;
    small.hc_lr = 16;
    small.n_expert = 8;
    small.n_ff = 62;

    for (int which = 0; which < 2; ++which) {
        const ModelGeometry& gg = which ? small : g;
        const char* tag = which ? "small" : "real";
        GdnBuffers gb;
        MoEBuffers mb;
        QsaBuffers qb;
        BlockBuffers bb;
        show_layout("gdn", tag, gdn_buffers_bytes(gg), gdn_buffers_init(gg, BASE, gb));
        show_layout("moe", tag, moe_buffers_bytes(gg, 8), moe_buffers_init(gg, 8, BASE, mb));
        show_layout("qsa", tag, qsa_buffers_bytes(gg, 4096), qsa_buffers_init(gg, 4096, BASE, qb));
        show_layout("block", tag, block_buffers_bytes(gg), block_buffers_init(gg, BASE, bb));
        std::printf("DUMP_STRIDE|%s|%llu\n", tag, (unsigned long long) dump_stride_floats(gg));
    }

    // the layouts field by field, real geometry
    {
        GdnBuffers gb;
        gdn_buffers_init(g, BASE, gb);
        off("gdn.x_q8k", gb.x_q8k);
        off("gdn.x_q8_0", gb.x_q8_0);
        off("gdn.x_bf16", gb.x_bf16);
        off("gdn.qkv", gb.qkv);
        off("gdn.alpha", gb.alpha);
        off("gdn.o", gb.o);
        off("gdn.y_q8k", gb.y_q8k);
        off("gdn.state", gb.state);
        off("gdn.conv_state", gb.conv_state);
        MoEBuffers mb;
        moe_buffers_init(g, 8, BASE, mb);
        off("moe.x_bf16", mb.x_bf16);
        off("moe.logits", mb.logits);
        off("moe.ids", mb.ids);
        off("moe.shared", mb.shared);
        off("moe.sh_scratch", mb.sh_scratch);
        off("moe.x_q8_0", mb.x_q8_0);
        off("moe.x_q8k", mb.x_q8k);
        QsaBuffers qb;
        qsa_buffers_init(g, 4096, BASE, qb);
        off("qsa.x_q8k", qb.x_q8k);
        off("qsa.x_bf16", qb.x_bf16);
        off("qsa.q_full", qb.q_full);
        off("qsa.idx_raw", qb.idx_raw);
        off("qsa.cell_scores", qb.cell_scores);
        off("qsa.ids", qb.ids);
        off("qsa.k_scratch", qb.k_scratch);
        off("qsa.attn", qb.attn);
        off("qsa.attn16", qb.attn16);
        off("qsa.attn_q8k", qb.attn_q8k);
        off("qsa.attn_scratch", qb.attn_scratch);
        BlockBuffers bb;
        block_buffers_init(g, BASE, bb);
        off("block.R", bb.R);
        off("block.mixed", bb.mixed);
        off("block.inject", bb.inject);
        off("block.inject2", bb.inject2);
        off("block.gr_rs", bb.gr_rs);
        off("block.head_q8k", bb.head_q8k);
        off("block.gr.xn", bb.gr.xn);
        off("block.gr.xq", bb.gr.xq);
        off("block.gr.lq", bb.gr.lq);
        off("block.gr.gated", bb.gr.gated);
        off("block.gr.lo", bb.gr.lo);
        std::printf("GR_BYTES|%zu\n", bb.gr.bytes);
    }

    // the size helpers
    for (int64_t n_ff : {640LL, 62LL, 0LL, 31LL})
        std::printf("SH_EXP|%lld|%llu\n", (long long) n_ff,
                    (unsigned long long) kernels::shared_expert_scratch_bytes(n_ff));
    for (int64_t cap : {(int64_t) kernels::qsa_selection_width(kernels::kTopkMaxCells, s), (int64_t) 64, (int64_t) 1})
        std::printf("QSA_DECODE_SCRATCH|%lld|%llu\n", (long long) cap,
                    (unsigned long long) kernels::qsa_decode_attn_scratch_floats(cap, s));
    for (int fmt : {0, 1, 2})
        std::printf("KV_BLOCK|%d|%llu\n", fmt, (unsigned long long) kernels::kv_block_bytes(s, fmt));
    for (int64_t slots : {1024LL, 1LL})
        std::printf("KV_MAP|%lld|%llu\n", (long long) slots,
                    (unsigned long long) kernels::kv_stream_map_bytes(slots));
    std::printf("QSA_STEP|%llu\n", (unsigned long long) kernels::qsa_step_bytes());
    std::printf("SEL_WIDTH|%lld\n", (long long) kernels::qsa_selection_width(kernels::kTopkMaxCells, s));

    // kv_plan: the env switches are read once per process through statics, so each row needs its own process.
    // The harness re-execs itself with the env set and a tag; see plan().
    struct Row {
        const char* tag;
        int64_t max_cells, ring, resident;
        bool ring_off, main_off;
    };
    const Row rows[] = {
        {"resident0_ring-1", 4096, -1, 0, false, false},
        {"resident0_ring1024", 4096, 1024, 0, false, false},
        {"resident4096_ring-1", 4096, -1, 4096, false, false},
        {"resident4096_ring0", 4096, 0, 4096, false, false},
        {"resident4096_ring0_min", 4096, 0, 100, false, false},
        {"resident4096_ring0_off", 4096, 0, 4096, false, true},
        {"resident4096_ring1024", 4096, 1024, 4096, false, false},
        {"resident4096_ring1024_off", 4096, 1024, 4096, true, false},
        {"resident4096_ring4096", 4096, 4096, 4096, false, false},
        {"resident4096_ring4092", 4096, 4092, 4096, false, false},
        {"big_resident4096_ring-1", 100000, -1, 4096, false, false},
        {"big_resident4096_ring-1_off", 100000, -1, 4096, false, true},
        {"big_resident100_ring-1", 100000, -1, 100, false, false},
        {"big_resident20480_ring1024", 100000, 1024, 20480, false, false},
        {"big_resident20480_ring1024_off", 100000, 1024, 20480, true, false},
        {"big_resident4096_ring0", 100000, 0, 4096, false, false},
        {"big_resident4096_ring0_off", 100000, 0, 4096, false, true},
        {"big_resident100_ring0", 100000, 0, 100, false, false},
    };
    for (const auto& r : rows) {
        std::string cmd = std::string("CORPUS_PLAN_ONLY=") + r.tag;
        cmd += " CORPUS_MAX=" + std::to_string(r.max_cells);
        cmd += " CORPUS_RING=" + std::to_string(r.ring);
        cmd += " CORPUS_RESIDENT=" + std::to_string(r.resident);
        if (r.ring_off) cmd += " CORPUS_RING_OFF=1";
        if (r.main_off) cmd += " CORPUS_MAIN_OFF=1";
        cmd += " " + self;
        FILE* p = popen(cmd.c_str(), "r");
        char buf[256];
        while (fgets(buf, sizeof buf, p)) fputs(buf, stdout);
        pclose(p);
    }

    // kv_pool_bytes and qsa_state_bytes across the modes
    for (int mode = 0; mode < 3; ++mode) {
        g_kv_int8 = mode == 1;
        g_kv_q4 = mode == 2;
        g_kv_hybrid = false;
        std::printf("KVPOOL|mode%d|plain=%llu\n", mode, (unsigned long long) kv_pool_bytes(s, 1024, false, false));
        std::printf("KVPOOL|mode%d|int8arg=%llu\n", mode, (unsigned long long) kv_pool_bytes(s, 1024, false, true));
        std::printf("KVPOOL|mode%d|hybrid=%llu\n", mode, (unsigned long long) kv_pool_bytes(s, 1024, true, false));
        g_kv_hybrid = true;
        std::printf("KVPOOL|mode%d|hybridflag=%llu\n", mode, (unsigned long long) kv_pool_bytes(s, 1024, true, true));
        for (int64_t ring : {-1LL, 1024LL}) {
            g_kv_resident = 4096;
            std::printf("QSA_STATE|mode%d|ring%lld|rope0=%llu|rope1=%llu\n", mode, (long long) ring,
                        (unsigned long long) qsa_state_bytes(g, 4096, false, ring),
                        (unsigned long long) qsa_state_bytes(g, 4096, true, ring));
        }
    }
    return 0;
}
