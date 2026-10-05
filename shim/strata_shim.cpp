// The C++ side of strata_kernels.h: a thin shim exposing real Strata code over
// the C ABI. Built two ways:
//   * any host (g++, no CUDA)  -> direct-IO slots backed by strata::platform::DirectFile
//   * a CUDA host (-DCUDA)     -> device/stream/graph/sampler slots too
// Slots a build cannot serve stay NULL and `strata_kernels_load` reports what
// was filled, so callers degrade instead of faulting.
#include "strata_kernels.h"

#include "strata/platform/direct_file.hpp"

#ifdef STRATA_SHIM_CUDA
#include <cuda_runtime.h>
#endif

#include <cstdio>
#include <cstring>
#include <new>

static_assert(STRATA_ABI_VERSION == 1u, "shim written for ABI v1");

namespace {

constexpr uint32_t kAbi = STRATA_ABI_VERSION;

void set_err(char* err, size_t err_len, const char* msg) {
    if (err != nullptr && err_len > 0) {
        std::snprintf(err, err_len, "%s", msg);
    }
}

// ---- direct-IO: a strata::platform::DirectFile behind the ABI handle ----
StrataStatus abi_file_open(const char* path, StrataDeviceFile* out, char* err, size_t err_len) {
    if (path == nullptr || out == nullptr) {
        set_err(err, err_len, "file_open: null path or out");
        return STRATA_ERROR;
    }
    auto* f = new (std::nothrow) strata::platform::DirectFile();
    if (f == nullptr) {
        set_err(err, err_len, "file_open: out of memory");
        return STRATA_ERROR;
    }
    std::string e;
    if (!f->open(path, e)) {
        delete f;
        set_err(err, err_len, e.c_str());
        return STRATA_ERROR;
    }
    *out = f;
    return STRATA_OK;
}

void abi_file_close(StrataDeviceFile file) {
    delete static_cast<strata::platform::DirectFile*>(file);
}

uint64_t abi_file_size(StrataDeviceFile file) {
    auto* f = static_cast<strata::platform::DirectFile*>(file);
    return f != nullptr ? f->size() : 0;
}

StrataStatus abi_file_submit(StrataDeviceFile file, uint64_t offset, void* buffer, uint32_t length,
                             uint64_t tag, char* err, size_t err_len) {
    auto* f = static_cast<strata::platform::DirectFile*>(file);
    if (f == nullptr) {
        set_err(err, err_len, "file_submit: null file");
        return STRATA_ERROR;
    }
    std::string e;
    if (!f->submit(offset, buffer, length, tag, e)) {
        set_err(err, err_len, e.c_str());
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

int32_t abi_file_wait(StrataDeviceFile file, IoCompletion* out, int32_t max, int32_t timeout_ms) {
    auto* f = static_cast<strata::platform::DirectFile*>(file);
    if (f == nullptr || out == nullptr || max <= 0) return 0;
    strata::platform::Completion comp[64];
    int total = 0;
    while (total < max) {
        const int want = max - total < 64 ? max - total : 64;
        const int got = f->wait(comp, want, timeout_ms);
        for (int i = 0; i < got; ++i) {
            out[total + i].tag = comp[i].tag;
            out[total + i].bytes = comp[i].bytes;
            out[total + i].ok = comp[i].ok ? 1u : 0u;
        }
        total += got;
        if (got < want) break;  // timed out or the queue drained
    }
    return total;
}

uint32_t abi_file_alignment(void) { return strata::platform::DirectFile::alignment(); }

void abi_pinned_free(void* p, uint64_t bytes) {
    if (p == nullptr) return;
#ifdef STRATA_SHIM_CUDA
    cudaFreeHost(p);
    (void) bytes;
#else
    strata::platform::DirectFile::free_aligned(p);
    (void) bytes;
#endif
}

StrataStatus abi_pinned_alloc(uint64_t bytes, void** out, char* err, size_t err_len) {
    if (out == nullptr || bytes == 0) {
        set_err(err, err_len, "pinned_alloc: null out or zero bytes");
        return STRATA_ERROR;
    }
#ifdef STRATA_SHIM_CUDA
    void* p = nullptr;
    const cudaError_t e = cudaHostAlloc(&p, (size_t) bytes, cudaHostAllocDefault);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    *out = p;
    return STRATA_OK;
#else
    void* p = strata::platform::DirectFile::alloc_aligned((size_t) bytes);
    if (p == nullptr) {
        set_err(err, err_len, "pinned_alloc: posix_memalign refused");
        return STRATA_ERROR;
    }
    *out = p;
    return STRATA_OK;
#endif
}

#ifdef STRATA_SHIM_CUDA
// ---------------------------------------------------------------- CUDA slots
int32_t abi_device_count(void) {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;
}

StrataStatus abi_device_info(int32_t ordinal, DeviceInfo* out) {
    if (out == nullptr) return STRATA_ERROR;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        (void) cudaGetLastError();
        return STRATA_ERROR;
    }
    std::memset(out, 0, sizeof *out);
    out->ordinal = ordinal;
    out->cc_major = p.major;
    out->cc_minor = p.minor;
    out->multi_processor_count = p.multiProcessorCount;
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    out->driver_version = driver;
    out->runtime_version = runtime;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        out->free_bytes = free_b;
        out->total_bytes = total_b;
    }
    std::snprintf(out->arch, sizeof out->arch, "sm_%d%d", p.major, p.minor);
    std::snprintf(out->name, sizeof out->name, "%s", p.name);
    return STRATA_OK;
}

StrataStatus abi_gpu_arch_problem(int32_t ordinal, char* out, size_t out_len) {
    (void) ordinal;
    if (out != nullptr && out_len > 0) out[0] = '\0';  // CUDA binaries are portable across arches via PTX/JIT
    return STRATA_OK;
}

StrataStatus abi_set_device(int32_t ordinal) {
    return cudaSetDevice(ordinal) == cudaSuccess ? STRATA_OK : STRATA_ERROR;
}

StrataStatus abi_stream_create(StrataStream* out) {
    if (out == nullptr) return STRATA_ERROR;
    cudaStream_t s{};
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return STRATA_ERROR;
    *out = (StrataStream) s;
    return STRATA_OK;
}

void abi_stream_destroy(StrataStream stream) {
    if (stream != nullptr) (void) cudaStreamDestroy((cudaStream_t) stream);
}

int32_t abi_stream_query_done(StrataStream stream) {
    if (stream == nullptr) return 1;
    const cudaError_t e = cudaStreamQuery((cudaStream_t) stream);
    if (e == cudaErrorNotReady) {
        (void) cudaGetLastError();  // a poll must consume its own error, or it lies later
        return 0;
    }
    return e == cudaSuccess ? 1 : 1;
}

struct GraphSlot {
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
};

// cudaGraphInstantiate dropped its pErrorNode argument in CUDA 12.
cudaError_t instantiate_graph(cudaGraphExec_t* exec, cudaGraph_t graph) {
#if CUDART_VERSION >= 12000
    return cudaGraphInstantiate(exec, graph, 0);
#else
    return cudaGraphInstantiate(exec, graph, nullptr, 0);
#endif
}

StrataStatus abi_graph_record(StrataGraph* out, StrataStream stream, char* err, size_t err_len) {
    if (out == nullptr || stream == nullptr) {
        set_err(err, err_len, "graph_record: null handle or stream");
        return STRATA_ERROR;
    }
    auto* slot = new (std::nothrow) GraphSlot();
    if (slot == nullptr) {
        set_err(err, err_len, "graph_record: out of memory");
        return STRATA_ERROR;
    }
    slot->stream = (cudaStream_t) stream;
    const cudaError_t e = cudaStreamBeginCapture(slot->stream, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) {
        delete slot;
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    *out = (StrataGraph) slot;
    return STRATA_OK;
}

StrataStatus abi_graph_end(StrataGraph graph, char* err, size_t err_len) {
    auto* slot = static_cast<GraphSlot*>(graph);
    if (slot == nullptr) {
        set_err(err, err_len, "graph_end: null handle");
        return STRATA_ERROR;
    }
    const cudaError_t e = cudaStreamEndCapture(slot->stream, &slot->graph);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    if (instantiate_graph(&slot->exec, slot->graph) != cudaSuccess) {
        set_err(err, err_len, "graph_end: instantiate failed");
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

StrataStatus abi_graph_replay(StrataGraph graph, StrataStream stream, char* err, size_t err_len) {
    auto* slot = static_cast<GraphSlot*>(graph);
    if (slot == nullptr || slot->exec == nullptr) {
        set_err(err, err_len, "graph_replay: graph not captured");
        return STRATA_ERROR;
    }
    if (cudaGraphLaunch(slot->exec, (cudaStream_t) stream) != cudaSuccess) {
        set_err(err, err_len, "graph_replay: launch failed");
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

void abi_graph_destroy(StrataGraph graph) {
    auto* slot = static_cast<GraphSlot*>(graph);
    if (slot == nullptr) return;
    if (slot->exec != nullptr) cudaGraphExecDestroy(slot->exec);
    if (slot->graph != nullptr) cudaGraphDestroy(slot->graph);
    delete slot;
}

StrataStatus abi_device_alloc(uint64_t bytes, void** out, char* err, size_t err_len) {
    if (out == nullptr || bytes == 0) {
        set_err(err, err_len, "device_alloc: null out or zero bytes");
        return STRATA_ERROR;
    }
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, (size_t) bytes);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    *out = p;
    return STRATA_OK;
}

void abi_device_free(void* p, uint64_t bytes) {
    if (p != nullptr) (void) cudaFree(p);
    (void) bytes;
}

StrataStatus abi_memcpy_h2d_async(void* dst_dev, const void* src_host, uint64_t bytes,
                                  StrataStream stream) {
    const cudaError_t e = cudaMemcpyAsync(dst_dev, src_host, (size_t) bytes, cudaMemcpyHostToDevice,
                                          (cudaStream_t) stream);
    return e == cudaSuccess ? STRATA_OK : STRATA_ERROR;
}

StrataStatus abi_memcpy_d2h_async(void* dst_host, const void* src_dev, uint64_t bytes,
                                  StrataStream stream) {
    const cudaError_t e = cudaMemcpyAsync(dst_host, src_dev, (size_t) bytes, cudaMemcpyDeviceToHost,
                                          (cudaStream_t) stream);
    return e == cudaSuccess ? STRATA_OK : STRATA_ERROR;
}

/* The snapshot contract's two synchronous slots. The engine core hands one
 * address per region and cudaMemcpyDefault resolves the direction, which is
 * exactly what the C++ engine did with raw pointers. */
StrataStatus abi_memcpy_default(void* dst, const void* src, uint64_t bytes,
                                char* err, size_t err_len) {
    if (bytes == 0) return STRATA_OK;
    const cudaError_t e = cudaMemcpy(dst, src, (size_t) bytes, cudaMemcpyDefault);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

StrataStatus abi_device_sync(char* err, size_t err_len) {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

/* ---- appended slots: pinned arena, pager, layer, session. Each forwards to the
 * runtime call of the same name and puts the runtime's own text in `err`, so the
 * Rust core prints exactly what the C++ printed. ---- */

StrataStatus abi_get_device(int32_t* out, char* err, size_t err_len) {
    if (out == nullptr) {
        set_err(err, err_len, "get_device: null out");
        return STRATA_ERROR;
    }
    int d = 0;
    const cudaError_t e = cudaGetDevice(&d);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    *out = (int32_t) d;
    return STRATA_OK;
}

/* cudaGetLastError CLEARS the sticky error; cudaPeekAtLastError does not. The
 * engine uses both and they are not interchangeable, so both are slots. */
StrataStatus abi_get_last_error(char* err, size_t err_len) {
    const cudaError_t e = cudaGetLastError();
    set_err(err, err_len, cudaGetErrorString(e));
    return e == cudaSuccess ? STRATA_OK : STRATA_ERROR;
}

StrataStatus abi_peek_last_error(char* err, size_t err_len) {
    const cudaError_t e = cudaPeekAtLastError();
    set_err(err, err_len, cudaGetErrorString(e));
    return e == cudaSuccess ? STRATA_OK : STRATA_ERROR;
}

StrataStatus abi_mem_get_info(uint64_t* free_bytes, uint64_t* total_bytes, char* err, size_t err_len) {
    size_t f = 0, t = 0;
    const cudaError_t e = cudaMemGetInfo(&f, &t);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    if (free_bytes != nullptr) *free_bytes = f;
    if (total_bytes != nullptr) *total_bytes = t;
    return STRATA_OK;
}

StrataStatus abi_host_register(void* host, uint64_t bytes, uint32_t flags, char* err, size_t err_len) {
    const cudaError_t e = cudaHostRegister(host, (size_t) bytes, flags);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

StrataStatus abi_host_unregister(void* host, char* err, size_t err_len) {
    const cudaError_t e = cudaHostUnregister(host);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

StrataStatus abi_host_get_device_pointer(void* host, void** out, uint32_t flags,
                                         char* err, size_t err_len) {
    if (out == nullptr) {
        set_err(err, err_len, "host_get_device_pointer: null out");
        return STRATA_ERROR;
    }
    void* p = nullptr;
    const cudaError_t e = cudaHostGetDevicePointer(&p, host, flags);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    *out = p;
    return STRATA_OK;
}

StrataStatus abi_memset_dev(void* dst_dev, int32_t value, uint64_t bytes, char* err, size_t err_len) {
    if (bytes == 0) return STRATA_OK;
    const cudaError_t e = cudaMemset(dst_dev, value, (size_t) bytes);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

StrataStatus abi_memset_async(void* dst_dev, int32_t value, uint64_t bytes, StrataStream stream,
                              char* err, size_t err_len) {
    if (bytes == 0) return STRATA_OK;
    const cudaError_t e =
        cudaMemsetAsync(dst_dev, value, (size_t) bytes, (cudaStream_t) stream);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

StrataStatus abi_memcpy2d_async(void* dst_dev, uint64_t dpitch, const void* src_host, uint64_t spitch,
                                uint64_t width, uint64_t height, int32_t kind, StrataStream stream,
                                char* err, size_t err_len) {
    const cudaError_t e = cudaMemcpy2DAsync(dst_dev, (size_t) dpitch, src_host, (size_t) spitch,
                                            (size_t) width, (size_t) height, (cudaMemcpyKind) kind,
                                            (cudaStream_t) stream);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

StrataStatus abi_stream_create_with_flags(StrataStream* out, uint32_t flags, char* err, size_t err_len) {
    if (out == nullptr) {
        set_err(err, err_len, "stream_create_with_flags: null out");
        return STRATA_ERROR;
    }
    const cudaError_t e = cudaStreamCreateWithFlags((cudaStream_t*) out, flags);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

StrataStatus abi_stream_sync(StrataStream stream, char* err, size_t err_len) {
    const cudaError_t e = cudaStreamSynchronize((cudaStream_t) stream);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

StrataStatus abi_stream_wait_event(StrataStream stream, StrataEvent event, uint32_t flags,
                                   char* err, size_t err_len) {
    const cudaError_t e = cudaStreamWaitEvent((cudaStream_t) stream, (cudaEvent_t) event, flags);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

/* flags 0 is cudaEventCreate; the engine only ever passes 0 today, and the flag
 * is carried so a `cudaEventCreateWithFlags` caller does not need its own slot. */
StrataStatus abi_event_create(StrataEvent* out, uint32_t flags, char* err, size_t err_len) {
    if (out == nullptr) {
        set_err(err, err_len, "event_create: null out");
        return STRATA_ERROR;
    }
    const cudaError_t e = cudaEventCreateWithFlags((cudaEvent_t*) out, flags);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

void abi_event_destroy(StrataEvent event) {
    if (event != nullptr) (void) cudaEventDestroy((cudaEvent_t) event);
}

StrataStatus abi_event_record(StrataEvent event, StrataStream stream, char* err, size_t err_len) {
    const cudaError_t e = cudaEventRecord((cudaEvent_t) event, (cudaStream_t) stream);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

StrataStatus abi_event_sync(StrataEvent event, char* err, size_t err_len) {
    const cudaError_t e = cudaEventSynchronize((cudaEvent_t) event);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

int32_t abi_event_query_done(StrataEvent event) {
    return cudaEventQuery((cudaEvent_t) event) == cudaSuccess ? 1 : 0;
}

StrataStatus abi_event_elapsed_ms(float* out_ms, StrataEvent start, StrataEvent end,
                                  char* err, size_t err_len) {
    if (out_ms == nullptr) {
        set_err(err, err_len, "event_elapsed_ms: null out");
        return STRATA_ERROR;
    }
    const cudaError_t e = cudaEventElapsedTime(out_ms, (cudaEvent_t) start, (cudaEvent_t) end);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}

/* Uploads the instantiated exec, which is the handle the engine holds. */
StrataStatus abi_graph_upload(StrataGraph graph, StrataStream stream, char* err, size_t err_len) {
    auto* slot = static_cast<GraphSlot*>(graph);
    if (slot == nullptr || slot->exec == nullptr) {
        set_err(err, err_len, "graph_upload: graph not captured");
        return STRATA_ERROR;
    }
    const cudaError_t e = cudaGraphUpload(slot->exec, (cudaStream_t) stream);
    if (e != cudaSuccess) {
        set_err(err, err_len, cudaGetErrorString(e));
        return STRATA_ERROR;
    }
    return STRATA_OK;
}
#endif  // STRATA_SHIM_CUDA

}  // namespace

StrataStatus strata_kernels_load(uint32_t abi_version, StrataKernels* out, char* err, size_t err_len) {
    if (out == nullptr) return STRATA_ERROR;
    if (abi_version != kAbi) {
        set_err(err, err_len, "shim is ABI v1");
        return STRATA_UNSUPPORTED;
    }
    std::memset(out, 0, sizeof *out);
    out->file_open = abi_file_open;
    out->file_close = abi_file_close;
    out->file_size = abi_file_size;
    out->file_submit = abi_file_submit;
    out->file_wait = abi_file_wait;
    out->file_alignment = abi_file_alignment;
    out->pinned_alloc = abi_pinned_alloc;
    out->pinned_free = abi_pinned_free;
#ifdef STRATA_SHIM_CUDA
    out->device_count = abi_device_count;
    out->device_info = abi_device_info;
    out->gpu_arch_problem = abi_gpu_arch_problem;
    out->set_device = abi_set_device;
    out->stream_create = abi_stream_create;
    out->stream_destroy = abi_stream_destroy;
    out->stream_query_done = abi_stream_query_done;
    out->graph_record = abi_graph_record;
    out->graph_end = abi_graph_end;
    out->graph_replay = abi_graph_replay;
    out->graph_destroy = abi_graph_destroy;
    out->device_alloc = abi_device_alloc;
    out->device_free = abi_device_free;
    out->memcpy_h2d_async = abi_memcpy_h2d_async;
    out->memcpy_d2h_async = abi_memcpy_d2h_async;
    out->memcpy_default = abi_memcpy_default;
    out->device_sync = abi_device_sync;
    out->get_device = abi_get_device;
    out->get_last_error = abi_get_last_error;
    out->peek_last_error = abi_peek_last_error;
    out->mem_get_info = abi_mem_get_info;
    out->host_register = abi_host_register;
    out->host_unregister = abi_host_unregister;
    out->host_get_device_pointer = abi_host_get_device_pointer;
    out->memset_dev = abi_memset_dev;
    out->memset_async = abi_memset_async;
    out->memcpy2d_async = abi_memcpy2d_async;
    out->stream_create_with_flags = abi_stream_create_with_flags;
    out->stream_sync = abi_stream_sync;
    out->stream_wait_event = abi_stream_wait_event;
    out->event_create = abi_event_create;
    out->event_destroy = abi_event_destroy;
    out->event_record = abi_event_record;
    out->event_sync = abi_event_sync;
    out->event_query_done = abi_event_query_done;
    out->event_elapsed_ms = abi_event_elapsed_ms;
    out->graph_upload = abi_graph_upload;
    /* sampler + coupled-draft slots land in Phase 2 alongside the kernel calls
     * they forward to; a shim build that lacks them leaves them NULL per the
     * append-only contract, and the Rust core falls back. */
#endif
    return STRATA_OK;
}
