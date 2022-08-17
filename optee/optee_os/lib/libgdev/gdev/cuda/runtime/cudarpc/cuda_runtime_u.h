#ifndef CUDA_RUNTIME_U_H__
#define CUDA_RUNTIME_U_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include "rpc/rpc.h"
#include "tee_internal_api.h" /* for sgx_satus_t etc. */

#include "ocelot/cuda/interface/cuda_runtime.h"

#include <stdlib.h> /* for size_t */

#define TEE_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif


cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaMemcpyToHost(void* dst, void* src, size_t count);
cudaError_t cudaMemcpyToDevice(void* dst, void* src, size_t count);
cudaError_t cudaLaunchKernelByName(char* funcname, int func_len, dim3 gridDim, dim3 blockDim, void* argbuf, int argbufsize, uint32_t* parameters, int partotal_size, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaFuncGetParametersByName(uint32_t* n_par, uint32_t* parameters, const char* entryname, int name_len);
cudaError_t cudaThreadSynchronize();
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetLastError();
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
cudaError_t cudaGetDeviceCount(int* count);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
