
#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>

typedef void* (*device_alloc)(size_t size, int cached);
typedef void  (*device_free)(void* buf);
typedef uint32_t (*device_virt_to_phys)(void* buf);
typedef uint32_t (*device_exec)(uint32_t insn_phy_addr,
                 uint32_t insn_count, uint32_t wait_cycles);

int main() {
    void* handle = dlopen("libvta_device.so", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        return 0;
    }
    device_alloc dalloc = (device_alloc)dlsym(handle, "VTAMemAlloc");
    device_free  dfree = (device_free)dlsym(handle, "VTADeviceFree");
    device_virt_to_phys  virt_phys = (device_virt_to_phys)dlsym(handle, "VTAMemGetPhyAddr");
    device_exec  dexec = (device_exec)dlsym(handle, "VTADeviceRun");
    void* buf = dalloc(0x10000, 0);
    uint32_t paddr = virt_phys(buf + 0x1000);
    fprintf(stderr, "addr %x %x\n", buf, paddr);
    dexec(0, 0, 0);
}