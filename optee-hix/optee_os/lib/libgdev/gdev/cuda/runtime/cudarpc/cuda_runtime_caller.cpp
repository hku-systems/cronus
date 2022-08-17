

#include <assert.h>

#include <ocelot/cuda/interface/cuda_runtime.h>
#include <ocelot/cuda/interface/CudaRuntimeInterface.h>

#define NOT_IMPLEMENTED fprintf(stderr, "%s not implemented\n", __FUNCTION__)
#define NOT_IMPLEMENTED_RET(x) \
	fprintf(stderr, "%s not implemented\n", __FUNCTION__); \
	return x;

void *dummy_handle = NULL;

void** __cudaRegisterFatBinary(void *fatCubin) {
	NOT_IMPLEMENTED_RET(&dummy_handle);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
	NOT_IMPLEMENTED;
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
	char *deviceAddress, const char *deviceName, int ext, int size,
	int constant, int global) {
	NOT_IMPLEMENTED;
}

void __cudaRegisterTexture(
        void **fatCubinHandle,
  const struct textureReference *hostVar,
  const void **deviceAddress,
  const char *deviceName,
        int dim,
        int norm,
        int ext) {
	NOT_IMPLEMENTED;
}

void __cudaRegisterShared(
  void **fatCubinHandle,
  void **devicePtr) {

}

void __cudaRegisterSharedVar(
  void **fatCubinHandle,
  void **devicePtr,
  size_t size,
  size_t alignment,
  int storage) {
	NOT_IMPLEMENTED;
}

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
	NOT_IMPLEMENTED;
}

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

extern "C" void __cudaRegisterFatBinaryEnd(
        void **fatCubinHandle
) { fprintf(stderr, "register fat bin\n"); }

extern char __cudaInitModule(
        void **fatCubinHandle
) { fprintf(stderr, "load module\n"); }

cudaError_t  cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMalloc3DArray(struct cudaArray** arrayPtr, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaMalloc(void **devPtr, size_t size) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMallocHost(void **ptr, size_t size) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaFree(void *devPtr) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaFreeHost(void *ptr) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaFreeArray(struct cudaArray *array) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}


cudaError_t  cudaHostAlloc(void **pHost, size_t bytes, unsigned int flags) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaHostRegister(void *pHost, size_t bytes, unsigned int flags) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaHostUnregister(void *pHost) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
	NOT_IMPLEMENTED;
}

cudaError_t  cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
	NOT_IMPLEMENTED;
}

cudaError_t  cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
	NOT_IMPLEMENTED;
}

cudaError_t  cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
	NOT_IMPLEMENTED;
}

cudaError_t  cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaMemset(void *devPtr, int value, size_t count) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaGetSymbolAddress(void **devPtr, const char *symbol) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaGetSymbolSize(size_t *size, const char *symbol) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaGetDeviceCount(int *count) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaSetDevice(int device) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaGetDevice(int *device) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaSetValidDevices(int *device_arr, int len) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaSetDeviceFlags( int flags ) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaBindTexture2D(size_t *offset,const struct textureReference *texref,const void *devPtr, const struct cudaChannelFormatDesc *desc,size_t width, size_t height, size_t pitch) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaUnbindTexture(const struct textureReference *texref) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaGetTextureReference(const struct textureReference **texref, const char *symbol) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

struct cudaChannelFormatDesc  cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f) {
	struct cudaChannelFormatDesc desc = {x, y, z, w, f};
	return desc;
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaGetLastError(void) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaPeekAtLastError() {
	NOT_IMPLEMENTED_RET(cudaSuccess);
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

#undef _CASE

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaSetupArgument(const void *arg, size_t size, size_t offset) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaLaunch(const char *entry) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache cacheConfig) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaStreamCreate(cudaStream_t *pStream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaStreamDestroy(cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaStreamSynchronize(cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaStreamQuery(cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaEventCreate(cudaEvent_t *event) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaEventQuery(cudaEvent_t event) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaEventSynchronize(cudaEvent_t event) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaEventDestroy(cudaEvent_t event) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaSetDoubleForDevice(double *d) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaSetDoubleForHost(double *d) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t cudaDeviceReset(void) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaDeviceSynchronize(void) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *c) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache c) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaThreadExit(void) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaThreadSynchronize(void) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaDriverGetVersion(int *driverVersion) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

cudaError_t  cudaRuntimeGetVersion(int *runtimeVersion) {
	NOT_IMPLEMENTED_RET(cudaSuccess);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
extern "C" {

void __cudaMutexOperation(int lock) {
	NOT_IMPLEMENTED;
}

int __cudaSynchronizeThreads(void** one, void* two) {
	NOT_IMPLEMENTED_RET(0);
}

void __cudaTextureFetch(const void* tex, void* index, int integer, void* val) {
	NOT_IMPLEMENTED;
}

}



