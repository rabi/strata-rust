// tools/layout_corpus.cpp — the spec harness for src/core/layout.cpp, which has NO dedicated
// C++ test (it is exercised only through the load path on real packs). This builds the real
// layout.cpp object, supplies the one undefined symbol (WeightTable::find) with an in-memory
// table, and:
//   1. asserts a battery of good/bad cases against the REAL engine code (the Rust port's spec),
//   2. dumps one line per case ("ok" or the exact error string) for the Rust test suite to
//      byte-compare against.
//
//   g++ -std=c++20 -Iinclude tools/layout_corpus.cpp src/core/layout.cpp -o /tmp/layout_corpus
//   /tmp/layout_corpus            # assertions + corpus on stdout
#include "strata/core/layout.hpp"

#include <cstdio>
#include <map>
#include <string>

// The one symbol layout.cpp needs from the loader, backed by an in-memory table.
std::map<std::string, strata::core::WeightRef> g_table;
const strata::core::WeightTable g_empty_table;

namespace strata::core {
const WeightRef* WeightTable::find(const std::string& name) const {
    auto it = ::g_table.find(name);
    return it == ::g_table.end() ? nullptr : &it->second;
}
}  // namespace strata::core

namespace {

strata::core::WeightRef w2(int64_t ne0, int64_t ne1) {
    strata::core::WeightRef r;
    r.ne0 = ne0;
    r.ne1 = ne1;
    r.bytes = (uint64_t) ne0 * (uint64_t) ne1 * 2;
    r.kind = strata::core::WeightKind::Verbatim;
    return r;
}
strata::core::WeightRef w1(int64_t elements, strata::core::WeightKind kind) {
    strata::core::WeightRef r;
    r.ne0 = elements;
    r.ne1 = 1;
    r.elements = elements;
    r.kind = kind;
    r.bytes = (uint64_t) elements * (kind == strata::core::WeightKind::Bf16InF32 ? 2u : 4u);
    return r;
}

using G = strata::core::ModelGeometry;
using K = strata::core::WeightKind;

// A pack with every tensor the checks require, at the geometry's own dimensions.
void pack_everything(const G& g) {
    g_table.clear();
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        g_table[p + "hc_attn_down.weight"] = w2(g.hc_dim(), g.hc_lr);
        g_table[p + "hc_attn_up.weight"] = w2(g.hc_lr, g.hc_dim());
        g_table[p + "hc_attn_inject.weight"] = w2(g.hc_dim(), g.hc);
        g_table[p + "hc_ffn_down.weight"] = w2(g.hc_dim(), g.hc_lr);
        g_table[p + "hc_ffn_up.weight"] = w2(g.hc_lr, g.hc_dim());
        g_table[p + "hc_ffn_inject.weight"] = w2(g.hc_dim(), g.hc);
        g_table[p + "ffn_gate_inp.weight"] = w2(g.n_embd, g.n_expert);
        g_table[p + "ffn_gate_shexp.weight"] = w2(g.n_embd, g.n_ff);
        g_table[p + "ffn_up_shexp.weight"] = w2(g.n_embd, g.n_ff);
        g_table[p + "ffn_down_shexp.weight"] = w2(g.n_ff, g.n_embd);
        g_table[p + "hc_attn_norm.weight"] = w1(g.hc_dim(), K::F32);
        g_table[p + "hc_ffn_norm.weight"] = w1(g.hc_dim(), K::F32);
        g_table[p + "ffn_gate_inp_shexp.weight"] = w1(g.n_embd, K::Bf16InF32);
        if (l % g.qsa_interval != g.qsa_interval - 1) {  // GDN
            g_table[p + "attn_qkv.weight"] = w2(g.n_embd, g.ssm_conv_channels);
            g_table[p + "attn_gate.weight"] = w2(g.n_embd, g.ssm_value_dim);
            g_table[p + "ssm_out.weight"] = w2(g.ssm_value_dim, g.n_embd);
            g_table[p + "ssm_conv1d.weight"] = w2(g.ssm_d_conv, g.ssm_conv_channels);
            g_table[p + "ssm_alpha.weight"] = w2(g.n_embd, g.ssm_v_heads);
            g_table[p + "ssm_beta.weight"] = w2(g.n_embd, g.ssm_v_heads);
            g_table[p + "ssm_a"] = w1(g.ssm_v_heads, K::F32);
            g_table[p + "ssm_dt.bias"] = w1(g.ssm_v_heads, K::F32);
            g_table[p + "ssm_norm.weight"] = w1(g.ssm_state_size, K::F32);
        } else {  // QSA
            g_table[p + "attn_q.weight"] = w2(g.n_embd, 2 * g.n_head * g.head_dim);
            g_table[p + "attn_k.weight"] = w2(g.n_embd, g.n_head_kv * g.head_dim);
            g_table[p + "attn_v.weight"] = w2(g.n_embd, g.n_head_kv * g.head_dim);
            g_table[p + "attn_output.weight"] = w2(g.n_head * g.head_dim, g.n_embd);
            g_table[p + "indexer.q_proj.weight"] = w2(g.n_embd, g.idx_q_heads * g.idx_key_dim);
            g_table[p + "indexer.k_proj.weight"] = w2(g.n_embd, g.idx_key_dim);
            g_table[p + "attn_q_norm.weight"] = w1(g.head_dim, K::F32);
            g_table[p + "attn_k_norm.weight"] = w1(g.head_dim, K::F32);
            g_table[p + "indexer.q_norm.weight"] = w1(g.idx_key_dim, K::F32);
            g_table[p + "indexer.k_norm.weight"] = w1(g.idx_key_dim, K::F32);
        }
    }
}

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-72s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

std::string run_layer(const G& g, int64_t layer) {
    std::string err;
    return strata::core::check_layer(g_empty_table, g, layer, err) ? "ok" : err;
}
std::string run_all(const G& g) {
    std::string err;
    return strata::core::check_all(g_empty_table, g, err) ? "ok" : err;
}

void emit(const char* name, const std::string& result) {
    std::printf("CORPUS %s\t%s\n", name, result.c_str());
}

}  // namespace

int main() {
    const G g;  // defaults = the shipped geometry

    // ---- 1. the good pack
    pack_everything(g);
    check(run_all(g) == "ok", "complete pack at the default geometry passes check_all");
    check(run_layer(g, 0) == "ok" && run_layer(g, 3) == "ok", "a GDN and a QSA layer both pass");
    emit("good_all", run_all(g));
    emit("good_layer0", run_layer(g, 0));
    emit("good_layer3", run_layer(g, 3));

    // ---- 2. missing tensor: the head's own example. Drop indexer.k_proj -> the check must
    //         name it, and must NOT silently pass (that is how a 4x term hides).
    {
        auto save = g_table["blk.3.indexer.k_proj.weight"];
        g_table.erase("blk.3.indexer.k_proj.weight");
        const std::string e = run_layer(g, 3);
        check(e == "layer 3: missing blk.3.indexer.k_proj.weight", "missing indexer.k_proj names itself");
        emit("missing_kproj", e);
        check(run_all(g) != "ok", "check_all refuses the pack with the tensor missing");
        emit("missing_kproj_all", run_all(g));
        g_table["blk.3.indexer.k_proj.weight"] = save;
    }

    // ---- 3. the round-194 confusion: k_proj sized with the QUERY head count (4x too big)
    {
        auto save = g_table["blk.3.indexer.k_proj.weight"];
        g_table["blk.3.indexer.k_proj.weight"] = w2(g.n_embd, g.idx_q_heads * g.idx_key_dim);
        const std::string e = run_layer(g, 3);
        check(e == "layer 3: blk.3.indexer.k_proj.weight ne1 is 512, the kernels require 128",
              "k_proj with the query head count refused, has/want in the message");
        emit("kproj_query_sized", e);
        g_table["blk.3.indexer.k_proj.weight"] = save;
    }

    // ---- 4. axis order: down/up transposed the same way is a DIFFERENT tensor shape
    {
        auto save = g_table["blk.0.hc_attn_up.weight"];
        g_table["blk.0.hc_attn_up.weight"] = w2(g.hc_dim(), g.hc_lr);  // swapped
        const std::string e = run_layer(g, 0);
        check(e == "layer 0: blk.0.hc_attn_up.weight ne0 is 10240, the kernels require 320",
              "transposed hc_attn_up refused on ne0");
        emit("up_transposed", e);
        g_table["blk.0.hc_attn_up.weight"] = save;
    }

    // ---- 5. ne1 mismatch labels ne1
    {
        auto save = g_table["blk.0.ffn_gate_inp.weight"];
        g_table["blk.0.ffn_gate_inp.weight"] = w2(g.n_embd, 511);
        const std::string e = run_layer(g, 0);
        check(e == "layer 0: blk.0.ffn_gate_inp.weight ne1 is 511, the kernels require 512",
              "ne1 mismatch labelled ne1, not ne0");
        emit("ne1_mismatch", e);
        g_table["blk.0.ffn_gate_inp.weight"] = save;
    }

    // ---- 6. the shared-expert gate form: bytes are what make it a real check
    {
        auto save = g_table["blk.0.ffn_gate_inp_shexp.weight"];
        g_table["blk.0.ffn_gate_inp_shexp.weight"] = w1(g.n_embd, K::F32);  // the incident
        const std::string e = run_layer(g, 0);
        check(e == "layer 0: blk.0.ffn_gate_inp_shexp.weight is engine form 2 (10240 B), the kernels read it as form 1 (5120 B)",
              "shexp gate stored f32 refused with both byte counts");
        emit("shexp_gate_f32", e);
        g_table["blk.0.ffn_gate_inp_shexp.weight"] = save;
    }

    // ---- 7. element count on a 1-D
    {
        auto save = g_table["blk.0.ssm_norm.weight"];
        g_table["blk.0.ssm_norm.weight"] = w1(127, K::F32);
        const std::string e = run_layer(g, 0);
        check(e == "layer 0: blk.0.ssm_norm.weight elements is 127, the kernels require 128",
              "ssm_norm one element short refused on elements");
        emit("ssm_norm_short", e);
        g_table["blk.0.ssm_norm.weight"] = save;
    }

    // ---- 8. family crossing: GDN tensors on a QSA layer pass (unchecked there), and the
    //         REVERSE also passes per layer but check_all only sees it as the OTHER set missing
    {
        auto save = g_table["blk.3.attn_qkv.weight"];
        g_table["blk.3.attn_qkv.weight"] = w2(g.n_embd, 999);  // GDN tensor, wrong shape, on a QSA layer
        check(run_layer(g, 3) == "ok", "a wrong GDN tensor on a QSA layer is not checked there");
        g_table["blk.3.attn_qkv.weight"] = save;
    }

    // ---- 9. layer bounds
    emit("layer_neg1", run_layer(g, -1));
    emit("layer_48", run_layer(g, 48));
    check(run_layer(g, 47) == "ok", "the last layer is in range");

    // ---- 10. geometry variants a pack can actually ship: interval 5, n_layers not divisible
    {
        G g5 = g;
        g5.qsa_interval = 5;
        g5.n_layers = 25;
        pack_everything(g5);  // pack_everything follows the interval, so it stays consistent
        check(run_all(g5) == "ok", "interval 5 / 25 layers: 5 QSA + 20 GDN passes");
        emit("interval5_all", run_all(g5));
        pack_everything(g);  // back to the 48-layer pack built at the default interval
        G g4 = g;
        g4.n_layers = 47;  // truncates a real pack: layer 47 (QSA) is beyond the range
        check(run_all(g4) == "ok", "47 of the 48 layers checked, the 48th simply not checked");
        emit("truncated_47", run_all(g4));
    }

    std::printf(g_fail ? "layout_corpus: %d FAILED\n" : "layout_corpus: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
