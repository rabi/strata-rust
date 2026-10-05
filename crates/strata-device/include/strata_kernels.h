/* strata_kernels.h — the C ABI between the Rust engine core and the C++/CUDA
 * kernel + device layer. Hand-mirrored from crates/strata-device/src/abi.rs;
 * strata-device's layout tests pin the sizes, and the C++ shim is compiled
 * against this header, so a mismatch fails a build on one side or a test on the
 * other. Slot order is append-only: a shim that predates a slot leaves it NULL.
 *
 * Every *_dev buffer is a DEVICE pointer; *_host is pinned host memory. Passing
 * host memory where a device pointer is expected faults inside the kernel with
 * an illegal memory access reported by some later, unrelated synchronising call.
 */
#ifndef STRATA_KERNELS_H
#define STRATA_KERNELS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum StrataStatus {
    STRATA_OK = 0,
    STRATA_ERROR = 1,        /* the error buffer says why */
    STRATA_WOULD_BLOCK = 2,
    STRATA_UNSUPPORTED = 3,  /* this device-layer build has no such slot */
} StrataStatus;

/* Mirrors strata::kernels::SamplerParams (include/strata/kernels/sampler.hpp).
 * Anything that changes per round under graph capture is DATA in a device
 * buffer, never a kernel argument. Size pinned to 64 bytes by test. */
typedef struct SamplerParams {
    int32_t top_k;           /* 1..64 as given; 0 or >64 keep the widest list, 64 */
    float top_p;             /* 1.0 disables */
    float min_p;             /* 0 disables; keeps p >= min_p * p_max */
    float temperature;       /* <= 0 means greedy */
    int32_t min_keep;
    int32_t penalty_last_n;  /* 0 disables */
    float penalty_repeat;
    float penalty_freq;
    float penalty_present;
    uint64_t seed;           /* Philox is counter-based on (seed, token index) */
    uint64_t counter;        /* absolute draw index of row 0 */
    bool greedy;             /* padded to 8-byte alignment: total 64 */
} SamplerParams;

/* The plain-data ABI form of strata::core::DeviceInfo (std::strings replaced by
 * fixed NUL-terminated UTF-8 buffers). */
typedef struct DeviceInfo {
    int32_t ordinal;
    int32_t cc_major;
    int32_t cc_minor;
    int32_t multi_processor_count;
    int32_t driver_version;
    int32_t runtime_version;
    uint64_t total_bytes;    /* as reported by cudaMemGetInfo at query time */
    uint64_t free_bytes;
    char arch[64];           /* HIP gcnArchName without its feature suffix */
    char name[128];
} DeviceInfo;

/* platform/direct_file.hpp Completion. */
typedef struct IoCompletion {
    uint64_t tag;            /* the caller's tag from submit */
    uint32_t bytes;          /* short only at end of file */
    uint32_t ok;
} IoCompletion;

typedef void* StrataStream;
typedef void* StrataDeviceFile;
typedef void* StrataGraph;

typedef StrataStatus (*StrataCoupledStage)(const SamplerParams* mapped_params,
                                          const int32_t* mapped_hist,
                                          SamplerParams* params_dev, int32_t* ring_dev,
                                          int32_t cap, StrataStream stream);
typedef StrataStatus (*StrataCoupledDraftSample)(float* logits_dev, int32_t n_vocab,
                                                const int32_t* sub_to_id_dev,
                                                const int32_t* id_to_sub_dev, int32_t id_vocab,
                                                const SamplerParams* params_dev, int32_t* ring_dev,
                                                int32_t cap, int32_t j, const int32_t* step_rec_dev,
                                                void* scratch_dev, int32_t* out_id, float* out_prob,
                                                StrataStream stream);

typedef struct StrataKernels {
    /* ---- device / runtime ---- */
    int32_t (*device_count)(void);
    StrataStatus (*device_info)(int32_t ordinal, DeviceInfo* out);
    /* "" when the device can run this binary, else the reason (HIP arch guard) */
    StrataStatus (*gpu_arch_problem)(int32_t ordinal, char* out, size_t out_len);
    StrataStatus (*set_device)(int32_t ordinal);
    StrataStatus (*stream_create)(StrataStream* out);
    void (*stream_destroy)(StrataStream stream);
    /* polls an event, never a blocking sync: a thread that only reads memory
     * never makes the driver flush (core/graph.hpp, measured) */
    int32_t (*stream_query_done)(StrataStream stream);

    /* ---- direct-IO file (the expert pager) ---- */
    StrataStatus (*file_open)(const char* path, StrataDeviceFile* out, char* err, size_t err_len);
    void (*file_close)(StrataDeviceFile file);
    uint64_t (*file_size)(StrataDeviceFile file);
    /* buffer and offset aligned to file_alignment() */
    StrataStatus (*file_submit)(StrataDeviceFile file, uint64_t offset, void* buffer,
                                uint32_t length, uint64_t tag, char* err, size_t err_len);
    int32_t (*file_wait)(StrataDeviceFile file, IoCompletion* out, int32_t max, int32_t timeout_ms);
    uint32_t (*file_alignment)(void);

    /* ---- pinned memory (the expert arena) ---- */
    StrataStatus (*pinned_alloc)(uint64_t bytes, void** out, char* err, size_t err_len);
    void (*pinned_free)(void* p, uint64_t bytes);

    /* ---- captured graphs. Every buffer the body touches must exist before
     * record and never move: the registry holds the graph, not the memory. ---- */
    StrataStatus (*graph_record)(StrataGraph* out, StrataStream stream, char* err, size_t err_len);
    StrataStatus (*graph_end)(StrataGraph graph, char* err, size_t err_len);
    StrataStatus (*graph_replay)(StrataGraph graph, StrataStream stream, char* err, size_t err_len);
    void (*graph_destroy)(StrataGraph graph);

    /* ---- sampler ---- */
    /* history_dev: (n_tokens, history_len) int32, one row per token row, -1 in
     * unused slots; NULL + 0 when no penalties apply */
    StrataStatus (*sample_tokens)(const float* logits_dev, int32_t n_tokens, int32_t n_vocab,
                                  const int32_t* history_dev, int32_t history_len,
                                  const SamplerParams* params, int32_t* out_dev,
                                  StrataStream stream);
    /* 8-CTA cluster greedy (sm_90+ CUDA only); 0 = cannot run, caller falls back */
    int32_t (*sample_greedy_cluster)(const float* logits_dev, int32_t n_tokens, int32_t n_vocab,
                                     int32_t* out_dev, StrataStream stream);

    /* ---- coupled draft sampling (STRATA_SPEC_COUPLED). Device pointers
     * throughout; every per-round value comes from device memory so these can
     * be captured. ---- */
    size_t (*coupled_scratch_bytes)(int32_t n_vocab); /* 0: too wide, coupled cannot run */
    StrataCoupledStage coupled_stage;
    StrataCoupledDraftSample coupled_draft_sample;

    /* ---- device memory + DMA. Appended within ABI v1: the caller pre-zeroes
     * `out` (so an old shim whose memset covers only its own smaller sizeof
     * still leaves these NULL), and a shim without CUDA leaves them NULL. ---- */
    StrataStatus (*device_alloc)(uint64_t bytes, void** out, char* err, size_t err_len);
    void (*device_free)(void* p, uint64_t bytes);
    /* source may be pageable; only pinned sources actually run async, and the
     * copy is stream-ordered, so stream_query_done settles it */
    StrataStatus (*memcpy_h2d_async)(void* dst_dev, const void* src_host, uint64_t bytes,
                                     StrataStream stream);
    StrataStatus (*memcpy_d2h_async)(void* dst_host, const void* src_dev, uint64_t bytes,
                                     StrataStream stream);

    /* ---- conversation snapshot. Appended within ABI v1. Synchronous: the
     * snapshot contract is a park/resume path, never a decode step, so it uses
     * cudaMemcpyDefault (the engine resolves either direction from the single
     * pointer) and cudaDeviceSynchronize, exactly as conversation_state.cpp and
     * conversation_snapshot.cpp do. The engine core keeps every address opaque
     * and selects them by layer/region; these two slots are the only bytes the
     * snapshot contract moves. The residency hooks (kv_stream_reset,
     * kv_ring_restore) intentionally have no slots: until a build fills them,
     * the core rejects any image whose state needs residency work, so a session
     * that was never fully resident can never be admitted on a build that
     * cannot make it readable. ---- */
    StrataStatus (*memcpy_default)(void* dst, const void* src, uint64_t bytes,
                                   char* err, size_t err_len);
    StrataStatus (*device_sync)(char* err, size_t err_len);
} StrataKernels;

/* The shim's single entry point: fills `*out` with the vtable for this build
 * (NULLs for absent slots) and returns STRATA_OK, or STRATA_UNSUPPORTED with a
 * reason in `err` when the build cannot serve this ABI version.
 * `*out` is pre-zeroed by the caller; slots the build lacks stay NULL. */
StrataStatus strata_kernels_load(uint32_t abi_version, StrataKernels* out, char* err, size_t err_len);
#define STRATA_ABI_VERSION 1u

#ifdef __cplusplus
}
#endif

/* Pin the layouts against the Rust side (strata-device's abi tests assert the
 * same numbers). A C++ shim compiled against this header cannot then be linked
 * against a Rust core with a different layout. */
#if !(defined(__cplusplus) && __cplusplus >= 201103L) && !(defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
#error "strata_kernels.h needs C11 or C++11 for static assertions on the ABI layout"
#endif
#ifdef __cplusplus
#define STRATA_ALIGNOF(T) alignof(T)
#define STRATA_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#define STRATA_ALIGNOF(T) _Alignof(T)
#define STRATA_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif
STRATA_STATIC_ASSERT(sizeof(SamplerParams) == 64, "SamplerParams must stay 64 bytes (strata-device abi.rs)");
STRATA_STATIC_ASSERT(STRATA_ALIGNOF(SamplerParams) == 8, "SamplerParams must stay 8-byte aligned");
STRATA_STATIC_ASSERT(offsetof(SamplerParams, top_k) == 0 && offsetof(SamplerParams, seed) == 40
                   && offsetof(SamplerParams, counter) == 48 && offsetof(SamplerParams, greedy) == 56,
               "SamplerParams field order must match strata::kernels::SamplerParams");
STRATA_STATIC_ASSERT(sizeof(DeviceInfo) == 232, "DeviceInfo must stay 232 bytes");
STRATA_STATIC_ASSERT(offsetof(DeviceInfo, total_bytes) == 24 && offsetof(DeviceInfo, arch) == 40
                   && offsetof(DeviceInfo, name) == 104,
               "DeviceInfo field order must match the Rust DeviceInfo");
STRATA_STATIC_ASSERT(sizeof(IoCompletion) == 16, "IoCompletion must stay 16 bytes");
STRATA_STATIC_ASSERT(offsetof(IoCompletion, bytes) == 8 && offsetof(IoCompletion, ok) == 12, "IoCompletion field order");
STRATA_STATIC_ASSERT(sizeof(StrataKernels) == 30 * sizeof(void *), "StrataKernels is 30 vtable slots");
#endif /* STRATA_KERNELS_H */
