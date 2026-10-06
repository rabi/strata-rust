// The driver-API declarations the host-only corpus build needs, shaped like the
// real <cuda.h>. Only what expert_cache.cpp names.
//
// The corpus fake hands back real function pointers through
// cudaGetDriverEntryPointByVersion, so these signatures are the contract the fake
// has to satisfy: get one wrong and the harness compiles against a shape the real
// driver never has. Sizes are pinned in tools/cuda_stub_asserts.cpp.
#pragma once
#include "cuda_runtime.h"

#define CUDAAPI  /* __cdecl on Windows, empty here */

typedef unsigned long long CUdeviceptr;
typedef int CUdevice;
typedef enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_PERMITTED = 20,
    CUDA_ERROR_NOT_SUPPORTED = 801
} CUresult;
typedef unsigned long long CUmemGenericAllocationHandle;

typedef enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103
} CUdevice_attribute;

typedef enum CUmemAllocationGranularity_flags {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
} CUmemAllocationGranularity_flags;

typedef enum CUmemAllocationType { CU_MEM_ALLOCATION_TYPE_INVALID = 0, CU_MEM_ALLOCATION_TYPE_PINNED = 1 } CUmemAllocationType;
typedef enum CUmemAllocationHandleType { CU_MEM_HANDLE_TYPE_NONE = 0, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1 } CUmemAllocationHandleType;
typedef enum CUmemLocationType { CU_MEM_LOCATION_TYPE_INVALID = 0, CU_MEM_LOCATION_TYPE_DEVICE = 1 } CUmemLocationType;
typedef enum CUmemAccess_flags {
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
} CUmemAccess_flags;

typedef struct CUmemLocation_st {
    CUmemLocationType type;
    int id;
} CUmemLocation;

typedef struct CUmemAllocationProp_st {
    CUmemAllocationType type;
    CUmemAllocationHandleType requestedHandleTypes;
    CUmemLocation location;
    void* win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned char usage;
        unsigned char reserved[4];
    } allocFlags;
} CUmemAllocationProp;

typedef struct CUmemAccessDesc_st {
    CUmemLocation location;
    CUmemAccess_flags flags;
} CUmemAccessDesc;

extern "C" {
CUresult cuDeviceGet(CUdevice*, int);
CUresult cuDeviceGetAttribute(int*, CUdevice_attribute, CUdevice);
CUresult cuMemGetAllocationGranularity(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
CUresult cuMemAddressReserve(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
CUresult cuMemAddressFree(CUdeviceptr, size_t);
CUresult cuMemCreate(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
CUresult cuMemRelease(CUmemGenericAllocationHandle);
CUresult cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
CUresult cuMemUnmap(CUdeviceptr, size_t);
CUresult cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
}
