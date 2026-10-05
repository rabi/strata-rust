// Bodies for the CUDA runtime stub. The corpus harnesses compile the real Strata
// .cpp files host-only; none of these are CALLED by the functions under test, they
// exist so the link succeeds. Each one aborts if it is ever reached, so a harness
// that accidentally wanders into the device layer says so instead of reporting a
// plausible number.
#include "cuda_runtime.h"

#include <cstdio>
#include <cstdlib>

namespace {
void untouchable(const char* fn) {
    std::fprintf(stderr, "cudastub: %s was called - the corpus must not reach the device layer\n", fn);
    std::abort();
}
}  // namespace

extern "C" {
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) { untouchable("cudaMemcpy"); return cudaSuccess; }
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) { untouchable("cudaMemcpyAsync"); return cudaSuccess; }
cudaError_t cudaMemcpy2DAsync(void*, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t) { untouchable("cudaMemcpy2DAsync"); return cudaSuccess; }
cudaError_t cudaDeviceSynchronize(void) { untouchable("cudaDeviceSynchronize"); return cudaSuccess; }
cudaError_t cudaGetLastError(void) { untouchable("cudaGetLastError"); return cudaSuccess; }
cudaError_t cudaPeekAtLastError(void) { untouchable("cudaPeekAtLastError"); return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { untouchable("cudaGetErrorString"); return ""; }
cudaError_t cudaMalloc(void**, size_t) { untouchable("cudaMalloc"); return cudaSuccess; }
cudaError_t cudaFree(void*) { untouchable("cudaFree"); return cudaSuccess; }
cudaError_t cudaHostAlloc(void**, size_t, unsigned int) { untouchable("cudaHostAlloc"); return cudaSuccess; }
cudaError_t cudaFreeHost(void*) { untouchable("cudaFreeHost"); return cudaSuccess; }
cudaError_t cudaGetDeviceCount(int*) { untouchable("cudaGetDeviceCount"); return cudaSuccess; }
cudaError_t cudaSetDevice(int) { untouchable("cudaSetDevice"); return cudaSuccess; }
cudaError_t cudaGetDevice(int*) { untouchable("cudaGetDevice"); return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int) { untouchable("cudaGetDeviceProperties"); return cudaSuccess; }
cudaError_t cudaDriverGetVersion(int*) { untouchable("cudaDriverGetVersion"); return cudaSuccess; }
cudaError_t cudaRuntimeGetVersion(int*) { untouchable("cudaRuntimeGetVersion"); return cudaSuccess; }
cudaError_t cudaMemGetInfo(size_t*, size_t*) { untouchable("cudaMemGetInfo"); return cudaSuccess; }
cudaError_t cudaStreamCreate(cudaStream_t*) { untouchable("cudaStreamCreate"); return cudaSuccess; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned int) { untouchable("cudaStreamCreateWithFlags"); return cudaSuccess; }
cudaError_t cudaStreamDestroy(cudaStream_t) { untouchable("cudaStreamDestroy"); return cudaSuccess; }
cudaError_t cudaStreamQuery(cudaStream_t) { untouchable("cudaStreamQuery"); return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { untouchable("cudaStreamSynchronize"); return cudaSuccess; }
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int) { untouchable("cudaStreamWaitEvent"); return cudaSuccess; }
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { untouchable("cudaGraphLaunch"); return cudaSuccess; }
cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t) { untouchable("cudaGraphUpload"); return cudaSuccess; }
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) { untouchable("cudaStreamBeginCapture"); return cudaSuccess; }
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*) { untouchable("cudaStreamEndCapture"); return cudaSuccess; }
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned long long) { untouchable("cudaGraphInstantiate"); return cudaSuccess; }
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { untouchable("cudaGraphExecDestroy"); return cudaSuccess; }
cudaError_t cudaGraphDestroy(cudaGraph_t) { untouchable("cudaGraphDestroy"); return cudaSuccess; }
cudaError_t cudaMemset(void*, int, size_t) { untouchable("cudaMemset"); return cudaSuccess; }
cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t) { untouchable("cudaMemsetAsync"); return cudaSuccess; }
cudaError_t cudaEventCreate(cudaEvent_t*) { untouchable("cudaEventCreate"); return cudaSuccess; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned int) { untouchable("cudaEventCreateWithFlags"); return cudaSuccess; }
cudaError_t cudaEventDestroy(cudaEvent_t) { untouchable("cudaEventDestroy"); return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { untouchable("cudaEventRecord"); return cudaSuccess; }
cudaError_t cudaEventSynchronize(cudaEvent_t) { untouchable("cudaEventSynchronize"); return cudaSuccess; }
cudaError_t cudaEventQuery(cudaEvent_t) { untouchable("cudaEventQuery"); return cudaSuccess; }
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t) { untouchable("cudaEventElapsedTime"); return cudaSuccess; }
cudaError_t cudaHostRegister(void*, size_t, unsigned int) { untouchable("cudaHostRegister"); return cudaSuccess; }
cudaError_t cudaHostUnregister(void*) { untouchable("cudaHostUnregister"); return cudaSuccess; }
cudaError_t cudaHostGetDevicePointer(void**, void*, unsigned int) { untouchable("cudaHostGetDevicePointer"); return cudaSuccess; }
}
