#include "cuda_runtime_t.h"

#include "tee_internal_api.h"
#include "rpc/rpc.h"
#include <string.h> /* for memcpy etc */
#include <stdlib.h> /* for malloc/free etc */

typedef TEE_Result (*ecall_invoke_entry) (char* buffer);

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

typedef struct ms_cudaDeviceSynchronize_t {
	cudaError_t ms_retval;
} ms_cudaDeviceSynchronize_t;

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

static TEE_Result tee_cudaFree(char *buffer)
{
	ms_cudaFree_t* ms = TEE_CAST(ms_cudaFree_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaFree_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_devPtr = ms->ms_devPtr;

	RPC_SERVER_DEBUG("");

	ms->ms_retval = cudaFree(_tmp_devPtr);


	return status;
}

static TEE_Result tee_cudaMalloc(char *buffer)
{
	ms_cudaMalloc_t* ms = TEE_CAST(ms_cudaMalloc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMalloc_t);

	TEE_Result status = TEE_SUCCESS;
	void** _tmp_devPtr = TEE_CAST(void**, buffer_start + 0);
	size_t _len_devPtr = 8;
	void** _in_devPtr = NULL;

	RPC_SERVER_DEBUG("");

	if (_tmp_devPtr != NULL) {
		if ((_in_devPtr = (void**)malloc(_len_devPtr)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_devPtr, 0, _len_devPtr);
	}
	ms->ms_retval = cudaMalloc(_in_devPtr, ms->ms_size);
err:
	if (_in_devPtr) {
		memcpy(_tmp_devPtr, _in_devPtr, _len_devPtr);
		free(_in_devPtr);
	}

	return status;
}

static TEE_Result tee_cudaMemcpyToHost(char *buffer)
{
	ms_cudaMemcpyToHost_t* ms = TEE_CAST(ms_cudaMemcpyToHost_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyToHost_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_dst = _tmp_count;
	void* _in_dst = NULL;
	void* _tmp_src = ms->ms_src;

	RPC_SERVER_DEBUG("");

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	ms->ms_retval = cudaMemcpyToHost(_in_dst, _tmp_src, _tmp_count);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}

	return status;
}

static TEE_Result tee_cudaMemcpyToDevice(char *buffer)
{
	ms_cudaMemcpyToDevice_t* ms = TEE_CAST(ms_cudaMemcpyToDevice_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyToDevice_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_src = _tmp_count;
	void* _in_src = NULL;

	RPC_SERVER_DEBUG("");

	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy(_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpyToDevice(_tmp_dst, _in_src, _tmp_count);
err:
	if (_in_src) free(_in_src);

	return status;
}

static TEE_Result tee_cudaLaunchKernelByName(char *buffer)
{
	ms_cudaLaunchKernelByName_t* ms = TEE_CAST(ms_cudaLaunchKernelByName_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaLaunchKernelByName_t);

	TEE_Result status = TEE_SUCCESS;
	char* _tmp_funcname = TEE_CAST(char*, buffer_start + 0);
	int _tmp_func_len = ms->ms_func_len;
	size_t _len_funcname = _tmp_func_len;
	char* _in_funcname = NULL;
	void* _tmp_argbuf = TEE_CAST(void*, buffer_start + 0 + _tmp_func_len);
	int _tmp_argbufsize = ms->ms_argbufsize;
	size_t _len_argbuf = _tmp_argbufsize;
	void* _in_argbuf = NULL;
	uint32_t* _tmp_parameters = TEE_CAST(uint32_t*, buffer_start + 0 + _tmp_func_len + _tmp_argbufsize);
	int _tmp_partotal_size = ms->ms_partotal_size;
	size_t _len_parameters = _tmp_partotal_size;
	uint32_t* _in_parameters = NULL;

	RPC_SERVER_DEBUG("");

	if (_tmp_funcname != NULL) {
		_in_funcname = (char*)malloc(_len_funcname);
		if (_in_funcname == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy(_in_funcname, _tmp_funcname, _len_funcname);
	}
	if (_tmp_argbuf != NULL) {
		_in_argbuf = (void*)malloc(_len_argbuf);
		if (_in_argbuf == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy(_in_argbuf, _tmp_argbuf, _len_argbuf);
	}
	if (_tmp_parameters != NULL) {
		_in_parameters = (uint32_t*)malloc(_len_parameters);
		if (_in_parameters == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy(_in_parameters, _tmp_parameters, _len_parameters);
	}
	ms->ms_retval = cudaLaunchKernelByName(_in_funcname, _tmp_func_len, ms->ms_gridDim, ms->ms_blockDim, _in_argbuf, _tmp_argbufsize, _in_parameters, _tmp_partotal_size, ms->ms_sharedMem, ms->ms_stream);
err:
	if (_in_funcname) free(_in_funcname);
	if (_in_argbuf) free(_in_argbuf);
	if (_in_parameters) free(_in_parameters);

	return status;
}

static TEE_Result tee_cudaFuncGetParametersByName(char *buffer)
{
	ms_cudaFuncGetParametersByName_t* ms = TEE_CAST(ms_cudaFuncGetParametersByName_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaFuncGetParametersByName_t);

	TEE_Result status = TEE_SUCCESS;
	uint32_t* _tmp_n_par = TEE_CAST(uint32_t*, buffer_start + 0);
	size_t _len_n_par = 4;
	uint32_t* _in_n_par = NULL;
	uint32_t* _tmp_parameters = TEE_CAST(uint32_t*, buffer_start + 0 + 4);
	size_t _len_parameters = 80;
	uint32_t* _in_parameters = NULL;
	char* _tmp_entryname = TEE_CAST(char*, buffer_start + 0 + 4 + 80);
	int _tmp_name_len = ms->ms_name_len;
	size_t _len_entryname = _tmp_name_len;
	char* _in_entryname = NULL;

	RPC_SERVER_DEBUG("");

	if (_tmp_n_par != NULL) {
		if ((_in_n_par = (uint32_t*)malloc(_len_n_par)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_n_par, 0, _len_n_par);
	}
	if (_tmp_parameters != NULL) {
		if ((_in_parameters = (uint32_t*)malloc(_len_parameters)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_parameters, 0, _len_parameters);
	}
	if (_tmp_entryname != NULL) {
		_in_entryname = (char*)malloc(_len_entryname);
		if (_in_entryname == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_entryname, _tmp_entryname, _len_entryname);
	}
	ms->ms_retval = cudaFuncGetParametersByName(_in_n_par, _in_parameters, (const char*)_in_entryname, _tmp_name_len);
err:
	if (_in_n_par) {
		memcpy(_tmp_n_par, _in_n_par, _len_n_par);
		free(_in_n_par);
	}
	if (_in_parameters) {
		memcpy(_tmp_parameters, _in_parameters, _len_parameters);
		free(_in_parameters);
	}
	if (_in_entryname) free((void*)_in_entryname);

	return status;
}

static TEE_Result tee_cudaThreadSynchronize(char *buffer)
{
	ms_cudaThreadSynchronize_t* ms = TEE_CAST(ms_cudaThreadSynchronize_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaThreadSynchronize_t);

	TEE_Result status = TEE_SUCCESS;

	RPC_SERVER_DEBUG("");

	ms->ms_retval = cudaThreadSynchronize();


	return status;
}

static TEE_Result tee_cudaDeviceSynchronize(char *buffer)
{
	ms_cudaDeviceSynchronize_t* ms = TEE_CAST(ms_cudaDeviceSynchronize_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaDeviceSynchronize_t);

	TEE_Result status = TEE_SUCCESS;

	RPC_SERVER_DEBUG("");

	ms->ms_retval = cudaDeviceSynchronize();


	return status;
}

static TEE_Result tee_cudaGetLastError(char *buffer)
{
	ms_cudaGetLastError_t* ms = TEE_CAST(ms_cudaGetLastError_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaGetLastError_t);

	TEE_Result status = TEE_SUCCESS;

	RPC_SERVER_DEBUG("");

	ms->ms_retval = cudaGetLastError();


	return status;
}

static TEE_Result tee_cudaGetDeviceProperties(char *buffer)
{
	ms_cudaGetDeviceProperties_t* ms = TEE_CAST(ms_cudaGetDeviceProperties_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaGetDeviceProperties_t);

	TEE_Result status = TEE_SUCCESS;
	struct cudaDeviceProp* _tmp_prop = TEE_CAST(struct cudaDeviceProp*, buffer_start + 0);
	size_t _len_prop = 1 * sizeof(*_tmp_prop);
	struct cudaDeviceProp* _in_prop = NULL;

	RPC_SERVER_DEBUG("");

	if (_tmp_prop != NULL) {
		if ((_in_prop = (struct cudaDeviceProp*)malloc(_len_prop)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_prop, 0, _len_prop);
	}
	ms->ms_retval = cudaGetDeviceProperties(_in_prop, ms->ms_device);
err:
	if (_in_prop) {
		memcpy(_tmp_prop, _in_prop, _len_prop);
		free(_in_prop);
	}

	return status;
}

static TEE_Result tee_cudaGetDeviceCount(char *buffer)
{
	ms_cudaGetDeviceCount_t* ms = TEE_CAST(ms_cudaGetDeviceCount_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaGetDeviceCount_t);

	TEE_Result status = TEE_SUCCESS;
	int* _tmp_count = TEE_CAST(int*, buffer_start + 0);
	size_t _len_count = 1 * sizeof(*_tmp_count);
	int* _in_count = NULL;

	RPC_SERVER_DEBUG("");

	if (_tmp_count != NULL) {
		if ((_in_count = (int*)malloc(_len_count)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_count, 0, _len_count);
	}
	ms->ms_retval = cudaGetDeviceCount(_in_count);
err:
	if (_in_count) {
		memcpy(_tmp_count, _in_count, _len_count);
		free(_in_count);
	}

	return status;
}

const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv;} ecall_table[11];
} g_ecall_table = {
	11,
	{
		{(void*)(uintptr_t)tee_cudaFree, 0},
		{(void*)(uintptr_t)tee_cudaMalloc, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyToHost, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyToDevice, 0},
		{(void*)(uintptr_t)tee_cudaLaunchKernelByName, 0},
		{(void*)(uintptr_t)tee_cudaFuncGetParametersByName, 0},
		{(void*)(uintptr_t)tee_cudaThreadSynchronize, 0},
		{(void*)(uintptr_t)tee_cudaDeviceSynchronize, 0},
		{(void*)(uintptr_t)tee_cudaGetLastError, 0},
		{(void*)(uintptr_t)tee_cudaGetDeviceProperties, 0},
		{(void*)(uintptr_t)tee_cudaGetDeviceCount, 0},
	}
};

int rpc_dispatch(char* buffer)
{
	uint32_t cmd_id = *(uint32_t*)buffer;
	ecall_invoke_entry entry = TEE_CAST(ecall_invoke_entry, g_ecall_table.ecall_table[cmd_id].ecall_addr);
	return (*entry)(buffer + sizeof(uint32_t));
}
