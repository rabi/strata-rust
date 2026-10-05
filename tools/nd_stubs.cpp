// The corpus's own definition of fnv1a64 — the byte loop of
// src/core/pinned.cu:263, transcribed (pinned.cu is CUDA; it needs nvcc).
#include "strata/core/pinned.hpp"

namespace strata::core {
uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}
}  // namespace strata::core
