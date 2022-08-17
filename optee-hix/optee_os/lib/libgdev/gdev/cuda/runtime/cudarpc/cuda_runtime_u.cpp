#include "cuda_runtime_u.h"
#include <errno.h>

typedef struct ms_cudaFree_t {
	cudaError_t ms_retval;
	void* ms_devPtr;
} ms_cudaFree_t;

typedef struct ms_cudaMalloc_t {
	cudaError_t ms_retval;
	void** ms_devPtr;
	size_t ms_size;
} ms_cudaMalloc_t;

typedef struct ms_cudaMemcpyToHost_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_src;
	size_t ms_count;
} ms_cudaMemcpyToHost_t;

typedef struct ms_cudaMemcpyToDevice_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_src;
	size_t ms_count;
} ms_cudaMemcpyToDevice_t;

typedef struct ms_cudaLaunchKernelByName_t {
	cudaError_t ms_retval;
	char* ms_funcname;
	int ms_func_len;
	dim3 ms_gridDim;
	dim3 ms_blockDim;
	void* ms_argbuf;
	int ms_argbufsize;
	uint32_t* ms_parameters;
	int ms_partotal_size;
	size_t ms_sharedMem;
	cudaStream_t ms_stream;
} ms_cudaLaunchKernelByName_t;

typedef struct ms_cudaFuncGetParametersByName_t {
	cudaError_t ms_retval;
	uint32_t* ms_n_par;
	uint32_t* ms_parameters;
	char* ms_entryname;
	int ms_name_len;
} ms_cudaFuncGetParametersByName_t;

typedef struct ms_cudaThreadSynchronize_t {
	cudaError_t ms_retval;
} ms_cudaThreadSynchronize_t;

typedef struct ms_cudaGetLastError_t {
	cudaError_t ms_retval;
} ms_cudaGetLastError_t;

typedef struct ms_cudaGetDeviceProperties_t {
	cudaError_t ms_retval;
	struct cudaDeviceProp* ms_prop;
	int ms_device;
} ms_cudaGetDeviceProperties_t;

typedef struct ms_cudaGetDeviceCount_t {
	cudaError_t ms_retval;
	int* ms_count;
} ms_cudaGetDeviceCount_t;

cudaError_t cudaFree(void* devPtr)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaFree_t* ms = TEE_CAST(ms_cudaFree_t*, enclave_buffer);;
	

	ms->ms_devPtr = devPtr;
	bufsize = sizeof(ms_cudaFree_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(0, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMalloc_t* ms = TEE_CAST(ms_cudaMalloc_t*, enclave_buffer);;
	

	ms->ms_devPtr = devPtr;
	ms->ms_size = size;
	bufsize = sizeof(ms_cudaMalloc_t) + 8;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(1, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(devPtr, enclave_buffer + sizeof(ms_cudaMalloc_t), 8);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

cudaError_t cudaMemcpyToHost(void* dst, void* src, size_t count)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyToHost_t* ms = TEE_CAST(ms_cudaMemcpyToHost_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_src = src;
	ms->ms_count = count;
	bufsize = sizeof(ms_cudaMemcpyToHost_t) + count + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(2, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpyToHost_t), count);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

cudaError_t cudaMemcpyToDevice(void* dst, void* src, size_t count)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyToDevice_t* ms = TEE_CAST(ms_cudaMemcpyToDevice_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyToDevice_t) + 0, src, count);

	ms->ms_dst = dst;
	ms->ms_src = src;
	ms->ms_count = count;
	bufsize = sizeof(ms_cudaMemcpyToDevice_t) + 0 + count;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(3, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

cudaError_t cudaLaunchKernelByName(char* funcname, int func_len, dim3 gridDim, dim3 blockDim, void* argbuf, int argbufsize, uint32_t* parameters, int partotal_size, size_t sharedMem, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaLaunchKernelByName_t* ms = TEE_CAST(ms_cudaLaunchKernelByName_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaLaunchKernelByName_t), funcname, func_len);
	memcpy(enclave_buffer + sizeof(ms_cudaLaunchKernelByName_t) + func_len, argbuf, argbufsize);
	memcpy(enclave_buffer + sizeof(ms_cudaLaunchKernelByName_t) + func_len + argbufsize, parameters, partotal_size);

	ms->ms_funcname = funcname;
	ms->ms_func_len = func_len;
	ms->ms_gridDim = gridDim;
	ms->ms_blockDim = blockDim;
	ms->ms_argbuf = argbuf;
	ms->ms_argbufsize = argbufsize;
	ms->ms_parameters = parameters;
	ms->ms_partotal_size = partotal_size;
	ms->ms_sharedMem = sharedMem;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaLaunchKernelByName_t) + func_len + argbufsize + partotal_size;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(4, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

cudaError_t cudaFuncGetParametersByName(uint32_t* n_par, uint32_t* parameters, const char* entryname, int name_len)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaFuncGetParametersByName_t* ms = TEE_CAST(ms_cudaFuncGetParametersByName_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaFuncGetParametersByName_t) + 4 + 80, entryname, name_len);

	ms->ms_n_par = n_par;
	ms->ms_parameters = parameters;
	ms->ms_entryname = (char*)entryname;
	ms->ms_name_len = name_len;
	bufsize = sizeof(ms_cudaFuncGetParametersByName_t) + 4 + 80 + name_len;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(5, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(n_par, enclave_buffer + sizeof(ms_cudaFuncGetParametersByName_t), 4);
		memcpy(parameters, enclave_buffer + sizeof(ms_cudaFuncGetParametersByName_t) + 4, 80);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

cudaError_t cudaThreadSynchronize()
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaThreadSynchronize_t* ms = TEE_CAST(ms_cudaThreadSynchronize_t*, enclave_buffer);;
	

	bufsize = sizeof(ms_cudaThreadSynchronize_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(6, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

cudaError_t cudaGetLastError()
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaGetLastError_t* ms = TEE_CAST(ms_cudaGetLastError_t*, enclave_buffer);;
	

	bufsize = sizeof(ms_cudaGetLastError_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(7, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaGetDeviceProperties_t* ms = TEE_CAST(ms_cudaGetDeviceProperties_t*, enclave_buffer);;
	

	ms->ms_prop = prop;
	ms->ms_device = device;
	bufsize = sizeof(ms_cudaGetDeviceProperties_t) + 1 * sizeof(*prop);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(8, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(prop, enclave_buffer + sizeof(ms_cudaGetDeviceProperties_t), 1 * sizeof(*prop));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

cudaError_t cudaGetDeviceCount(int* count)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaGetDeviceCount_t* ms = TEE_CAST(ms_cudaGetDeviceCount_t*, enclave_buffer);;
	

	ms->ms_count = count;
	bufsize = sizeof(ms_cudaGetDeviceCount_t) + 1 * sizeof(*count);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(9, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(count, enclave_buffer + sizeof(ms_cudaGetDeviceCount_t), 1 * sizeof(*count));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

