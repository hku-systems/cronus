#include "vta_runtime_u.h"
#include <errno.h>

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

VTADeviceHandle VTADeviceAlloc()
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	VTADeviceHandle retval;
	ms_VTADeviceAlloc_t* ms = TEE_CAST(ms_VTADeviceAlloc_t*, enclave_buffer);;
	

	bufsize = sizeof(ms_VTADeviceAlloc_t);

	if(bufsize > buffer_size_in_bytes) {
		abort();
	}
	status = rpc_ecall(0, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

void VTADeviceFree(VTADeviceHandle handle)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	
	ms_VTADeviceFree_t* ms = TEE_CAST(ms_VTADeviceFree_t*, enclave_buffer);;
	

	ms->ms_handle = handle;
	bufsize = sizeof(ms_VTADeviceFree_t);

	if(bufsize > buffer_size_in_bytes) {
		abort();
	}
	status = rpc_ecall(1, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
	}

}

int VTADeviceRun(VTADeviceHandle device, vta_phy_addr_t insn_phy_addr, uint32_t insn_count, uint32_t wait_cycles)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	int retval;
	ms_VTADeviceRun_t* ms = TEE_CAST(ms_VTADeviceRun_t*, enclave_buffer);;
	

	ms->ms_device = device;
	ms->ms_insn_phy_addr = insn_phy_addr;
	ms->ms_insn_count = insn_count;
	ms->ms_wait_cycles = wait_cycles;
	bufsize = sizeof(ms_VTADeviceRun_t);

	if(bufsize > buffer_size_in_bytes) {
		abort();
	}
	status = rpc_ecall(2, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

void* VTAMemAlloc(size_t size, int cached)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	void* retval;
	ms_VTAMemAlloc_t* ms = TEE_CAST(ms_VTAMemAlloc_t*, enclave_buffer);;
	

	ms->ms_size = size;
	ms->ms_cached = cached;
	bufsize = sizeof(ms_VTAMemAlloc_t);

	if(bufsize > buffer_size_in_bytes) {
		abort();
	}
	status = rpc_ecall(3, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

void VTAMemFree(void* buf)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	
	ms_VTAMemFree_t* ms = TEE_CAST(ms_VTAMemFree_t*, enclave_buffer);;
	

	ms->ms_buf = buf;
	bufsize = sizeof(ms_VTAMemFree_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		abort();
	}
	status = rpc_ecall(4, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
	}

}

vta_phy_addr_t VTAMemGetPhyAddr(void* buf)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	vta_phy_addr_t retval;
	ms_VTAMemGetPhyAddr_t* ms = TEE_CAST(ms_VTAMemGetPhyAddr_t*, enclave_buffer);;
	

	ms->ms_buf = buf;
	bufsize = sizeof(ms_VTAMemGetPhyAddr_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		abort();
	}
	status = rpc_ecall(5, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}

}

void VTAMemCopyFromHost(void* dst, const void* src, size_t size)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	
	ms_VTAMemCopyFromHost_t* ms = TEE_CAST(ms_VTAMemCopyFromHost_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_VTAMemCopyFromHost_t) + 0, src, size);

	ms->ms_dst = dst;
	ms->ms_src = (void*)src;
	ms->ms_size = size;
	bufsize = sizeof(ms_VTAMemCopyFromHost_t) + 0 + size;

	if(bufsize > buffer_size_in_bytes) {
		abort();
	}
	status = rpc_ecall(6, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
	}

}

void VTAMemCopyToHost(void* dst, const void* src, size_t size)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	
	ms_VTAMemCopyToHost_t* ms = TEE_CAST(ms_VTAMemCopyToHost_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_src = (void*)src;
	ms->ms_size = size;
	bufsize = sizeof(ms_VTAMemCopyToHost_t) + size + 0;

	if(bufsize > buffer_size_in_bytes) {
		abort();
	}
	status = rpc_ecall(7, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		memcpy(dst, enclave_buffer + sizeof(ms_VTAMemCopyToHost_t), size);
	}

}

void VTAFlushCache(void* vir_addr, vta_phy_addr_t phy_addr, int size)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	
	ms_VTAFlushCache_t* ms = TEE_CAST(ms_VTAFlushCache_t*, enclave_buffer);;
	

	ms->ms_vir_addr = vir_addr;
	ms->ms_phy_addr = phy_addr;
	ms->ms_size = size;
	bufsize = sizeof(ms_VTAFlushCache_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		abort();
	}
	status = rpc_ecall(8, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
	}

}

void VTAInvalidateCache(void* vir_addr, vta_phy_addr_t phy_addr, int size)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	
	ms_VTAInvalidateCache_t* ms = TEE_CAST(ms_VTAInvalidateCache_t*, enclave_buffer);;
	

	ms->ms_vir_addr = vir_addr;
	ms->ms_phy_addr = phy_addr;
	ms->ms_size = size;
	bufsize = sizeof(ms_VTAInvalidateCache_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		abort();
	}
	status = rpc_ecall(9, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
	}

}

