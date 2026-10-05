// Bodies for strata::platform, for the same reason as cudastub_defs.cpp: the
// corpus links the real sources but never calls these. Abort if reached.
#include "strata/platform/memory.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::platform {

namespace {
void untouchable(const char* fn) {
    std::fprintf(stderr, "pstub: %s was called - the corpus must not reach the platform layer\n", fn);
    std::abort();
}
}  // namespace

LockResult lock_resident(void* p, uint64_t bytes) {
    untouchable("lock_resident");
    (void) p;
    (void) bytes;
    return {};
}

void unlock_resident(void* p, uint64_t bytes) {
    untouchable("unlock_resident");
    (void) p;
    (void) bytes;
}

}  // namespace strata::platform
