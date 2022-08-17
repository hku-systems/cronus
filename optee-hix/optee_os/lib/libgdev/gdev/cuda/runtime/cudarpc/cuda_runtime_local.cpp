
#include <assert.h>
#include <stdio.h>

#include <ocelot/cuda/interface/cuda_runtime.h>

#include "cuda_runtime_u.h"

dim3 pushed_gridDim;
dim3 pushed_blockDim;
size_t pushed_sharedMem;
struct CUstream_st *pushed_stream;

extern "C" unsigned __cudaPushCallConfiguration(
        dim3 gridDim,
        dim3 blockDim,
        size_t sharedMem,
        struct CUstream_st *stream) {
	pushed_gridDim = gridDim;
	pushed_blockDim = blockDim;
	pushed_sharedMem = sharedMem;
	pushed_stream = stream;
	return cudaSuccess;
}

extern "C" cudaError_t __cudaPopCallConfiguration(
        dim3 *gridDim,
        dim3 *blockDim,
        size_t *sharedMem,
        void *stream
) {
	cudaStream_t *__stream = (cudaStream_t*)stream;
	*gridDim = pushed_gridDim;
	*blockDim = pushed_blockDim;
	*sharedMem = pushed_sharedMem;
	*__stream = (cudaStream_t)(long)pushed_stream;
	return cudaSuccess;
}

void *dummy_handle = NULL;
void** __cudaRegisterFatBinary(void *fatCubin) {
    fprintf(stderr, "register fat bin\n");
    return &dummy_handle;
}

extern "C" void __cudaRegisterFatBinaryEnd(
        void **fatCubinHandle
) {
    fprintf(stderr, "register fat bin\n"); 
}

extern char __cudaInitModule(
        void **fatCubinHandle
) { fprintf(stderr, "load module\n"); }

#define MAX_FUNCS 50

static char* func_names[MAX_FUNCS];
static const char* func_ptr[MAX_FUNCS];
static int func_cnt = 0;

void __cudaRegisterFunction(
        void **fatCubinHandle,
  const char *hostFun,
        char *deviceFun,
  const char *deviceName,
        int thread_limit,
        uint3 *tid,
        uint3 *bid,
        dim3 *bDim,
        dim3 *gDim,
        int *wSize) {
	func_names[func_cnt] = deviceFun;
	func_ptr[func_cnt] = hostFun;
        func_cnt += 1;
}

#define KERNEL_ARGC_SIZE 64

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
        char *func_name = NULL;
        cudaError_t ret;
        uint32_t n_par;
        uint32_t parameters[20];
        uint32_t total_parameter_sizes = 0;
        void* args_copy = NULL;
        int args_copy_offset = 0;

        for (int i = 0;i < func_cnt;i++) {
                if (func_ptr[i] == func) {
                        func_name = func_names[i];
                }
        }
        if (!func_name) {
                return cudaErrorLaunchFailure;
        }

        ret = cudaFuncGetParametersByName(&n_par, parameters, func_name, strlen(func_name) + 1);

        if (ret) {
                return ret;
        }

        for (int i = 0;i < n_par; i++)
                total_parameter_sizes += parameters[i];
        
        args_copy = malloc(total_parameter_sizes);

        for (int i = 0;i < n_par;i++) {
                memcpy(args_copy + args_copy_offset, args[i], parameters[i]);
                args_copy_offset += parameters[i];
        }

        ret = cudaLaunchKernelByName(func_name, strlen(func_name) + 1, gridDim, blockDim, args_copy, total_parameter_sizes, parameters, sizeof(uint32_t) * n_par, sharedMem, stream);

        free(args_copy);

        return ret;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
	fprintf(stderr, "UnregisterFatBinary\n");
}

cudaError_t  cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
	if (kind == cudaMemcpyHostToDevice) {
        return cudaMemcpyToDevice(dst, (void*) src, count);
    } else if (kind == cudaMemcpyDeviceToHost) {
        return cudaMemcpyToHost(dst, (void*) src, count);
    } else {
        return cudaErrorUnknown;
    }
}

#define _CASE(x) case x: return #x;

const char*  cudaGetErrorString(cudaError_t error) {
	switch (error) {
	_CASE(cudaSuccess)
	_CASE(cudaErrorMissingConfiguration)
	_CASE(cudaErrorMemoryAllocation)
	_CASE(cudaErrorInitializationError)
	_CASE(cudaErrorLaunchFailure)
	_CASE(cudaErrorPriorLaunchFailure)
	_CASE(cudaErrorLaunchTimeout)
	_CASE(cudaErrorLaunchOutOfResources)
	_CASE(cudaErrorInvalidDeviceFunction)
	_CASE(cudaErrorInvalidConfiguration)
	_CASE(cudaErrorInvalidDevice)
	_CASE(cudaErrorInvalidValue)
	_CASE(cudaErrorInvalidPitchValue)
	_CASE(cudaErrorInvalidSymbol)
	_CASE(cudaErrorMapBufferObjectFailed)
	_CASE(cudaErrorUnmapBufferObjectFailed)
	_CASE(cudaErrorInvalidHostPointer)
	_CASE(cudaErrorInvalidDevicePointer)
	_CASE(cudaErrorInvalidTexture)
	_CASE(cudaErrorInvalidTextureBinding)
	_CASE(cudaErrorInvalidChannelDescriptor)
	_CASE(cudaErrorInvalidMemcpyDirection)
	_CASE(cudaErrorAddressOfConstant)
	_CASE(cudaErrorTextureFetchFailed)
	_CASE(cudaErrorTextureNotBound)
	_CASE(cudaErrorSynchronizationError)
	_CASE(cudaErrorInvalidFilterSetting)
	_CASE(cudaErrorInvalidNormSetting)
	_CASE(cudaErrorMixedDeviceExecution)
	_CASE(cudaErrorCudartUnloading)
	_CASE(cudaErrorUnknown)
	_CASE(cudaErrorNotYetImplemented)
	_CASE(cudaErrorMemoryValueTooLarge)
	_CASE(cudaErrorInvalidResourceHandle)
	_CASE(cudaErrorNotReady)
	_CASE(cudaErrorInsufficientDriver)
	_CASE(cudaErrorSetOnActiveProcess)
	_CASE(cudaErrorNoDevice)
	_CASE(cudaErrorStartupFailure)
	_CASE(cudaErrorApiFailureBase)
		default:
		break;
	}
	return "unimplemented";
}