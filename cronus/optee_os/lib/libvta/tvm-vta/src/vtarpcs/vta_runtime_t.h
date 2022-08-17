#ifndef VTA_RUNTIME_T_H__
#define VTA_RUNTIME_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>

#include "tee_internal_api.h"
#include "vta/driver.h"
#include "vta/hw_spec.h"

#include <stdlib.h> /* for size_t */

#define TEE_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif


VTADeviceHandle VTADeviceAlloc();
void VTADeviceFree(VTADeviceHandle handle);
int VTADeviceRun(VTADeviceHandle device, vta_phy_addr_t insn_phy_addr, uint32_t insn_count, uint32_t wait_cycles);
void* VTAMemAlloc(size_t size, int cached);
void VTAMemFree(void* buf);
vta_phy_addr_t VTAMemGetPhyAddr(void* buf);
void VTAMemCopyFromHost(void* dst, const void* src, size_t size);
void VTAMemCopyToHost(void* dst, const void* src, size_t size);
void VTAFlushCache(void* vir_addr, vta_phy_addr_t phy_addr, int size);
void VTAInvalidateCache(void* vir_addr, vta_phy_addr_t phy_addr, int size);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
