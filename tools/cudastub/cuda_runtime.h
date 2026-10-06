// Minimal CUDA runtime stub for host-only compilation of Strata sources.
//
// The corpus harnesses compile the real .cpp files on a machine with no CUDA
// toolkit: the declarations below exist so the code COMPILES, and the bodies are
// never called (the harness `--wrap`s the calls it actually exercises, and the
// ones it does not exercise are only referenced from code paths it never enters).
// Signatures are shaped like the real runtime; a wrong shape here is a compile
// error, which is the point.
#pragma once
#include <cstddef>
#include <cstdint>
typedef enum cudaError { cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorMemoryAllocation=2, cudaErrorUnknown=99 } cudaError_t;
constexpr cudaError_t cudaErrorNotReady = (cudaError_t) 600;
typedef enum cudaMemcpyKind { cudaMemcpyHostToHost=0, cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2, cudaMemcpyDeviceToDevice=3, cudaMemcpyDefault=4 } cudaMemcpyKind;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUgraph_st* cudaGraph_t;
typedef struct CUgraphExec_st* cudaGraphExec_t;
typedef struct CUgraphNode_st* cudaGraphNode_t;
extern "C" {
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetLastError(void);
#define cudaErrorNotReady_unused 1
#ifndef CUDART_VERSION
// 12050, not the older 12040 the stub started on: that puts the corpus on
// cudaGetDriverEntryPoint, and every CUDA deployment this engine runs on is 12.5+
// and takes cudaGetDriverEntryPointByVersion. The stub must exercise the path the
// product takes.
#define CUDART_VERSION 12050u
#endif
const char* cudaGetErrorString(cudaError_t);
}
/* Wider surface for the shim's compile check: signatures shaped like the real
 * runtime; bodies never run, the check is the point. */
extern "C" {
cudaError_t cudaMalloc(void**, size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaHostAlloc(void**, size_t, unsigned int);
cudaError_t cudaFreeHost(void*);
enum { cudaHostAllocDefault = 0, cudaHostAllocPortable = 1, cudaHostAllocMapped = 2 };
cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDevice(int*);
typedef struct {
    int major, minor;
    int multiProcessorCount;
    size_t totalGlobalMem, sharedMemPerBlock;
    int regsPerBlock, warpSize, maxThreadsPerBlock;
    char name[256];
    int gcnArch, gcnArchMajor, gcnArchMinor, gcnArchPatch;
    char gcnArchName[256];
} cudaDeviceProp;
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
cudaError_t cudaDriverGetVersion(int*);
cudaError_t cudaRuntimeGetVersion(int*);
cudaError_t cudaMemGetInfo(size_t*, size_t*);
enum { cudaStreamNonBlocking = 1 };
cudaError_t cudaStreamCreate(cudaStream_t*);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned int);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamQuery(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t);
enum cudaStreamCaptureMode { cudaStreamCaptureModeThreadLocal = 1 };
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaMemset(void*, int, size_t);
cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t);
/* events and the pinned-registration path: what the appended ABI slots forward
 * to, and what pinned.cu and expert_source.cpp call directly. */
enum { cudaEventDefault = 0, cudaEventBlockingSync = 1, cudaEventDisableTiming = 2 };
cudaError_t cudaEventCreate(cudaEvent_t*);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned int);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
cudaError_t cudaPeekAtLastError(void);
enum { cudaHostRegisterDefault = 0, cudaHostRegisterPortable = 1, cudaHostRegisterMapped = 2 };
cudaError_t cudaHostRegister(void*, size_t, unsigned int);
cudaError_t cudaHostUnregister(void*);
cudaError_t cudaHostGetDevicePointer(void**, void*, unsigned int);
cudaError_t cudaMemcpy2DAsync(void*, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind,
                              cudaStream_t);

// The driver entry-point lookup expert_cache.cpp uses to reach the VMM functions
// without linking the driver library. The stub's fake hands back real pointers to
// its own functions through here, so this is the seam the harness drives.
enum cudaDriverEntryPointQueryResult {
    cudaDriverEntryPointSuccess = 0,
    cudaDriverEntryPointSymbolNotFound = 1,
    cudaDriverEntryPointVersionNotSufficent = 2
};
enum cudaGetDriverEntryPointFlags { cudaEnableDefault = 0, cudaEnableLegacyStream = 1, cudaEnablePerThreadDefaultStream = 2 };
cudaError_t cudaGetDriverEntryPoint(const char*, void**, unsigned long long, cudaDriverEntryPointQueryResult*);
cudaError_t cudaGetDriverEntryPointByVersion(const char*, void**, unsigned int, cudaGetDriverEntryPointFlags,
                                             cudaDriverEntryPointQueryResult*);

// The real header has template overloads for the void** allocators.
}  // extern "C"

template <class T> cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**) p, n); }
template <class T> cudaError_t cudaHostAlloc(T** p, size_t n, unsigned int f) { return cudaHostAlloc((void**) p, n, f); }

