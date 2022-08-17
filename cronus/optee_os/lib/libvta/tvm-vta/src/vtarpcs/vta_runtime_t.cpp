#include "vta_runtime_t.h"

#include "tee_internal_api.h"
#include "rpc/rpc.h"
#include <string.h> /* for memcpy etc */
#include <stdlib.h> /* for malloc/free etc */

typedef TEE_Result (*ecall_invoke_entry) (char* buffer);

typedef struct ms_VTADeviceAlloc_t {
	VTADeviceHandle ms_retval;
} ms_VTADeviceAlloc_t;

typedef struct ms_VTADeviceFree_t {
	VTADeviceHandle ms_handle;
} ms_VTADeviceFree_t;

typedef struct ms_VTADeviceRun_t {
	int ms_retval;
	VTADeviceHandle ms_device;
	vta_phy_addr_t ms_insn_phy_addr;
	uint32_t ms_insn_count;
	uint32_t ms_wait_cycles;
} ms_VTADeviceRun_t;

typedef struct ms_VTAMemAlloc_t {
	void* ms_retval;
	size_t ms_size;
	int ms_cached;
} ms_VTAMemAlloc_t;

typedef struct ms_VTAMemFree_t {
	void* ms_buf;
} ms_VTAMemFree_t;

typedef struct ms_VTAMemGetPhyAddr_t {
	vta_phy_addr_t ms_retval;
	void* ms_buf;
} ms_VTAMemGetPhyAddr_t;

typedef struct ms_VTAMemCopyFromHost_t {
	void* ms_dst;
	void* ms_src;
	size_t ms_size;
} ms_VTAMemCopyFromHost_t;

typedef struct ms_VTAMemCopyToHost_t {
	void* ms_dst;
	void* ms_src;
	size_t ms_size;
} ms_VTAMemCopyToHost_t;

typedef struct ms_VTAFlushCache_t {
	void* ms_vir_addr;
	vta_phy_addr_t ms_phy_addr;
	int ms_size;
} ms_VTAFlushCache_t;

typedef struct ms_VTAInvalidateCache_t {
	void* ms_vir_addr;
	vta_phy_addr_t ms_phy_addr;
	int ms_size;
} ms_VTAInvalidateCache_t;

static TEE_Result tee_VTADeviceAlloc(char *buffer)
{
	ms_VTADeviceAlloc_t* ms = TEE_CAST(ms_VTADeviceAlloc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_VTADeviceAlloc_t);

	TEE_Result status = TEE_SUCCESS;

	RPC_SERVER_DEBUG("");

	ms->ms_retval = VTADeviceAlloc();


	return status;
}

static TEE_Result tee_VTADeviceFree(char *buffer)
{
	ms_VTADeviceFree_t* ms = TEE_CAST(ms_VTADeviceFree_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_VTADeviceFree_t);

	TEE_Result status = TEE_SUCCESS;

	RPC_SERVER_DEBUG("");

	VTADeviceFree(ms->ms_handle);


	return status;
}

static TEE_Result tee_VTADeviceRun(char *buffer)
{
	ms_VTADeviceRun_t* ms = TEE_CAST(ms_VTADeviceRun_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_VTADeviceRun_t);

	TEE_Result status = TEE_SUCCESS;

	RPC_SERVER_DEBUG("");

	ms->ms_retval = VTADeviceRun(ms->ms_device, ms->ms_insn_phy_addr, ms->ms_insn_count, ms->ms_wait_cycles);


	return status;
}

static TEE_Result tee_VTAMemAlloc(char *buffer)
{
	ms_VTAMemAlloc_t* ms = TEE_CAST(ms_VTAMemAlloc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_VTAMemAlloc_t);

	TEE_Result status = TEE_SUCCESS;

	RPC_SERVER_DEBUG("");

	ms->ms_retval = VTAMemAlloc(ms->ms_size, ms->ms_cached);


	return status;
}

static TEE_Result tee_VTAMemFree(char *buffer)
{
	ms_VTAMemFree_t* ms = TEE_CAST(ms_VTAMemFree_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_VTAMemFree_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_buf = ms->ms_buf;

	RPC_SERVER_DEBUG("");

	VTAMemFree(_tmp_buf);


	return status;
}

static TEE_Result tee_VTAMemGetPhyAddr(char *buffer)
{
	ms_VTAMemGetPhyAddr_t* ms = TEE_CAST(ms_VTAMemGetPhyAddr_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_VTAMemGetPhyAddr_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_buf = ms->ms_buf;

	RPC_SERVER_DEBUG("");

	ms->ms_retval = VTAMemGetPhyAddr(_tmp_buf);


	return status;
}

static TEE_Result tee_VTAMemCopyFromHost(char *buffer)
{
	ms_VTAMemCopyFromHost_t* ms = TEE_CAST(ms_VTAMemCopyFromHost_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_VTAMemCopyFromHost_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + 0);
	size_t _tmp_size = ms->ms_size;
	size_t _len_src = _tmp_size;
	void* _in_src = NULL;

	RPC_SERVER_DEBUG("");

	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	VTAMemCopyFromHost(_tmp_dst, (const void*)_in_src, _tmp_size);
err:
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_VTAMemCopyToHost(char *buffer)
{
	ms_VTAMemCopyToHost_t* ms = TEE_CAST(ms_VTAMemCopyToHost_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_VTAMemCopyToHost_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_size = ms->ms_size;
	size_t _len_dst = _tmp_size;
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
	VTAMemCopyToHost(_in_dst, (const void*)_tmp_src, _tmp_size);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}

	return status;
}

static TEE_Result tee_VTAFlushCache(char *buffer)
{
	ms_VTAFlushCache_t* ms = TEE_CAST(ms_VTAFlushCache_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_VTAFlushCache_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_vir_addr = ms->ms_vir_addr;

	RPC_SERVER_DEBUG("");

	VTAFlushCache(_tmp_vir_addr, ms->ms_phy_addr, ms->ms_size);


	return status;
}

static TEE_Result tee_VTAInvalidateCache(char *buffer)
{
	ms_VTAInvalidateCache_t* ms = TEE_CAST(ms_VTAInvalidateCache_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_VTAInvalidateCache_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_vir_addr = ms->ms_vir_addr;

	RPC_SERVER_DEBUG("");

	VTAInvalidateCache(_tmp_vir_addr, ms->ms_phy_addr, ms->ms_size);


	return status;
}

const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv;} ecall_table[10];
} g_ecall_table = {
	10,
	{
		{(void*)(uintptr_t)tee_VTADeviceAlloc, 0},
		{(void*)(uintptr_t)tee_VTADeviceFree, 0},
		{(void*)(uintptr_t)tee_VTADeviceRun, 0},
		{(void*)(uintptr_t)tee_VTAMemAlloc, 0},
		{(void*)(uintptr_t)tee_VTAMemFree, 0},
		{(void*)(uintptr_t)tee_VTAMemGetPhyAddr, 0},
		{(void*)(uintptr_t)tee_VTAMemCopyFromHost, 0},
		{(void*)(uintptr_t)tee_VTAMemCopyToHost, 0},
		{(void*)(uintptr_t)tee_VTAFlushCache, 0},
		{(void*)(uintptr_t)tee_VTAInvalidateCache, 0},
	}
};

int rpc_dispatch(char* buffer)
{
	uint32_t cmd_id = *(uint32_t*)buffer;
	ecall_invoke_entry entry = TEE_CAST(ecall_invoke_entry, g_ecall_table.ecall_table[cmd_id].ecall_addr);
	return (*entry)(buffer + sizeof(uint32_t));
}
