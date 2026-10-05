// The three host-side functions of `include/strata/kernels/native_mmvq.hpp`,
// for the corpus binary. The engine defines them in
// `src/kernels/cuda/native_mmvq.cu`, which needs nvcc; the corpus runs on a box
// with no CUDA toolkit, so this file provides them — with the case mapping
// copied from the .cu and every byte count taken from `sizeof(block_*)` in the
// SAME header the kernel static_asserts against (`third_party/ggml/ggml-common.h`),
// so the numbers cannot drift from the kernel's own layout:
//
//   native_mmvq.cu:105-124  static_assert(sizeof(Q5KBlock)==176, Q81Block==36,
//                          Q20Block==18, Q3KBlock==110, IQ4XSBlock==136,
//                          Q4KBlock==144, Q6KBlock==210, Q40Block==18,
//                          Q50Block==22, Q80Block==34, IQ4NLBlock==18)
//   native_mmvq.cu:1565     native_mmvq_supported — the type set
//   native_mmvq.cu:1572     native_mmvq_weight_bytes — the (elems, bytes) table
//   iq_kernels.cu:1780      iq_row_bytes — the IQ types' per-256-element size
//
// `tools/native_dense_corpus.cpp` prints these functions' output as golden
// CORPUS|BYTES lines; a mismatch against a machine that links the real .cu is a
// real bug in one of the two, and the L4 host build is where that gets checked.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml-common.h"

#include <limits>
#include <stdexcept>

namespace strata::kernels {

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

/// `src/kernels/cuda/iq_kernels.cu:1780` — the same table, the same sizeofs.
size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t)(n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t)(n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t)(n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t)(n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t)(n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t)(n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t)(n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t)(n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t)(n / 256) * sizeof(block_q3_K);
        case 42: return (size_t)(n / 64) * sizeof(block_q2_0);
        case 12: return (size_t)(n / 256) * sizeof(block_q4_K);
        case 13: return (size_t)(n / 256) * sizeof(block_q5_K);
        case 7:  return (size_t)(n / 32) * sizeof(block_q5_1);
        case 6:  return (size_t)(n / 32) * sizeof(block_q5_0);
        case 8:  return (size_t)(n / 32) * sizeof(block_q8_0);
        case 30: return (size_t) n * 2;  // BF16
        default: return 0;
    }
}

namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int MAX_NCOLS = 8;

void validate_shape(int n_in, int ncols, int block_elems) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument(
            "native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS)
        throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}

}  // namespace

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols, Q8K);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(block_q8_1);
}

/// `sizeof` of the block struct this GGML type's rows are made of, plus its
/// element count. The corpus prints these so the Rust port's table is checked
/// against the header the kernel static_asserts against, rather than against a
/// transcription of it.
extern "C" void native_block_size(int ggml_type, int* block_elems, int* block_bytes) {
    switch (ggml_type) {
        case 2:  *block_elems = 32;  *block_bytes = (int) sizeof(block_q4_0);    break;
        case 6:  *block_elems = 32;  *block_bytes = (int) sizeof(block_q5_0);    break;
        case 7:  *block_elems = 32;  *block_bytes = (int) sizeof(block_q5_1);    break;
        case 8:  *block_elems = 32;  *block_bytes = (int) sizeof(block_q8_0);    break;
        case 20: *block_elems = 32;  *block_bytes = (int) sizeof(block_iq4_nl);  break;
        case 42: *block_elems = 64;  *block_bytes = (int) sizeof(block_q2_0);    break;
        case 11: *block_elems = 256; *block_bytes = (int) sizeof(block_q3_K);    break;
        case 12: *block_elems = 256; *block_bytes = (int) sizeof(block_q4_K);    break;
        case 13: *block_elems = 256; *block_bytes = (int) sizeof(block_q5_K);    break;
        case 14: *block_elems = 256; *block_bytes = (int) sizeof(block_q6_K);    break;
        case 23: *block_elems = 256; *block_bytes = (int) sizeof(block_iq4_xs);  break;
        case 16: *block_elems = 256; *block_bytes = (int) sizeof(block_iq2_xxs); break;
        case 17: *block_elems = 256; *block_bytes = (int) sizeof(block_iq2_xs);  break;
        case 18: *block_elems = 256; *block_bytes = (int) sizeof(block_iq3_xxs); break;
        case 21: *block_elems = 256; *block_bytes = (int) sizeof(block_iq3_s);   break;
        case 22: *block_elems = 256; *block_bytes = (int) sizeof(block_iq2_s);   break;
        case 29: *block_elems = 256; *block_bytes = (int) sizeof(block_iq1_m);   break;
        default: *block_elems = 0;   *block_bytes = -1;                          break;
    }
}

extern "C" int sizeof_block_q8_1() { return (int) sizeof(block_q8_1); }

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems = 0, block_bytes = 0;
    switch (ggml_type) {
        case 2: block_elems = 32;  block_bytes = (int) sizeof(block_q4_0);   break;
        case 6: block_elems = 32;  block_bytes = (int) sizeof(block_q5_0);   break;
        case 7: block_elems = 32;  block_bytes = (int) sizeof(block_q5_1);   break;
        case 8: block_elems = 32;  block_bytes = (int) sizeof(block_q8_0);   break;
        case 20: block_elems = 32; block_bytes = (int) sizeof(block_iq4_nl); break;
        case 11: block_elems = 256; block_bytes = (int) sizeof(block_q3_K);  break;
        case 12: block_elems = 256; block_bytes = (int) sizeof(block_q4_K);  break;
        case 13: block_elems = 256; block_bytes = (int) sizeof(block_q5_K);  break;
        case 14: block_elems = 256; block_bytes = (int) sizeof(block_q6_K);  break;
        case 23: block_elems = 256; block_bytes = (int) sizeof(block_iq4_xs); break;
        case 42: block_elems = 64;  block_bytes = (int) sizeof(block_q2_0);  break;
        case 16: block_elems = QK; block_bytes = (int) iq_row_bytes(16, QK); break;
        case 17: block_elems = QK; block_bytes = (int) iq_row_bytes(17, QK); break;
        case 18: block_elems = QK; block_bytes = (int) iq_row_bytes(18, QK); break;
        case 21: block_elems = QK; block_bytes = (int) iq_row_bytes(21, QK); break;
        case 22: block_elems = QK; block_bytes = (int) iq_row_bytes(22, QK); break;
        case 29: block_elems = QK; block_bytes = (int) iq_row_bytes(29, QK); break;
        default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out))
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    return row_bytes * std::size_t(n_out);
}

}  // namespace strata::kernels
