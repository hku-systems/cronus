
#include <assert.h>
#include <stdio.h>

#include <ocelot/cuda/interface/cuda_runtime.h>

#include "cuda_runtime_t.h"

cudaError_t cudaMemcpyToHost(void *dst, void *src, size_t count) {
    return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
}

cudaError_t cudaMemcpyToDevice(void *dst, void *src, size_t count) {
    return cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}

extern char* _kernel_func_names[];
extern const char* _kernel_func_ptrs[];
extern int _kernel_func_cnt;

cudaError_t cudaLaunchKernelByName(char* funcname, int func_len, dim3 gridDim, dim3 blockDim, 
    void* args_buf, int total_size, uint32_t* parameters, int par_len, size_t sharedMem, cudaStream_t stream) {
    const char* func_ptr = NULL;
    for (int i = 0;i < _kernel_func_cnt;i++) {
        if (strcmp(funcname, _kernel_func_names[i]) == 0) {
            func_ptr = _kernel_func_ptrs[i];
        }
    }
    if (!func_ptr) {
        return cudaErrorLaunchFailure;
    }

    int parameter_len = (int)(par_len / sizeof(uint32_t));
    void** args = (void**) malloc(parameter_len * sizeof(void*));
    int total_offset = 0;
    for (int i = 0;i < parameter_len;i++) {
        args[i] = args_buf + total_offset;
        total_offset += parameters[i];
    }

    return cudaLaunchKernel(func_ptr, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t cudaFuncGetParametersByName(uint32_t *n_par, uint32_t parameters[20], const char *funcname, int name_len) {
    const char* func_ptr = NULL;
    cudaError_t ret = cudaErrorLaunchFailure;

    for (int i = 0;i < _kernel_func_cnt;i++) {
        if (strcmp(funcname, _kernel_func_names[i]) == 0) {
            func_ptr = _kernel_func_ptrs[i];
        }
    }
    if (!func_ptr) {
        return cudaErrorLaunchFailure;
    }

    ret = cudaFuncGetParameters(n_par, parameters, func_ptr);

    return ret;
}