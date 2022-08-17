/*
 * QEMU vtacational PCI device
 *
 * Copyright (c) 2012-2015 Jiri Slaby
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <dlfcn.h>

#include "qemu/osdep.h"
#include "qemu/units.h"
#include "hw/pci/pci.h"
#include "hw/hw.h"
#include "hw/pci/msi.h"
#include "qemu/timer.h"
#include "qom/object.h"
#include "qemu/main-loop.h" /* iothread mutex */
#include "qemu/module.h"
#include "qapi/visitor.h"

#define TYPE_PCI_VTA_DEVICE "vta"
typedef struct EduState EduState;
DECLARE_INSTANCE_CHECKER(EduState, VTA,
                         TYPE_PCI_VTA_DEVICE)

#define FACT_IRQ        0x00000001
#define DMA_IRQ         0x00000100

#define DMA_START       0x40000
#define DMA_SIZE        4096

#define VTA_CTL_SIZE    (4096 * 16)
#define VTA_MEM_SIZE    ((vta->ram_size_gb * GiB))

/* Allows 32 concurrent tasks */
#define VTA_CMD_QUEUE_SIZE 32

typedef void* (*device_alloc)(size_t size, int cached);
typedef void  (*device_free)(void* buf);
typedef uint32_t (*device_virt_to_phys)(void* buf);
typedef uint32_t (*device_exec)(uint32_t insn_phy_addr, uint32_t offset, 
                 uint32_t insn_count, uint32_t wait_cycles);

typedef struct {
    union {
        struct {
            uint32_t insn_phy_addr;
            uint32_t insn_count;
            uint32_t wait_cycles;
            uint32_t offset;
            uint32_t status;
#define VTA_CTRL_STATUS_OFFSET 4
        };
        uint32_t data[5];
    };
} exec_t;

typedef struct {
    int task_cur_idx;
    int task_last_idx;
    int task_cnt;
    int task_idxs[VTA_CMD_QUEUE_SIZE];
    QemuMutex mutex;
    QemuCond cv;
} task_queue_t;

struct EduState {
    PCIDevice pdev;
    MemoryRegion mmio;
    MemoryRegion ram;
    MemoryRegion subram;

    QemuThread thread;
    QemuMutex thr_mutex;
    QemuCond thr_cond;

    bool stopping;

    uint32_t irq_status;

#define VTA_DMA_RUN             0x1
#define VTA_DMA_DIR(cmd)        (((cmd) & 0x2) >> 1)
# define VTA_DMA_FROM_PCI       0
# define VTA_DMA_TO_PCI         1
#define VTA_DMA_IRQ             0x4
    struct dma_state {
        dma_addr_t src;
        dma_addr_t dst;
        dma_addr_t cnt;
        dma_addr_t cmd;
    } dma;
    char dma_buf[DMA_SIZE];
    uint64_t dma_mask;

    // control struct
    exec_t main_ctl[VTA_CMD_QUEUE_SIZE];
    task_queue_t queue_ctl;

    uint8_t *mem;
    uint32_t ram_size_gb;

    // stubs to the backend device
    device_alloc alloc;
    device_free free;
    device_virt_to_phys virt_phys;
    device_exec exec;
};

static void queue_init(task_queue_t *q) {
    q->task_cur_idx = 0;
    q->task_last_idx = 0;
    q->task_cnt = 0;
    memset(q->task_idxs, 0, sizeof(q->task_idxs));
    qemu_mutex_init(&q->mutex);
    qemu_cond_init(&q->cv);
}

static void queue_push(task_queue_t *q, int idx) {
    qemu_mutex_lock(&q->mutex);
    q->task_idxs[q->task_cur_idx] = idx;
    q->task_cur_idx += 1;
    q->task_cur_idx = q->task_cur_idx % VTA_CMD_QUEUE_SIZE;
    q->task_cnt += 1;
    qemu_mutex_unlock(&q->mutex);
    qemu_cond_signal(&q->cv);
}

static int queue_pop(task_queue_t *q, bool *stoping) {
    qemu_mutex_lock(&q->mutex);
    while (!q->task_cnt && !*stoping) {
        qemu_cond_wait(&q->cv, &q->mutex);
    }
    int idx = q->task_idxs[q->task_last_idx];
    q->task_last_idx += 1;
    q->task_last_idx = q->task_last_idx % VTA_CMD_QUEUE_SIZE;
    q->task_cnt -= 1;
    qemu_mutex_unlock(&q->mutex);
    return idx;
}

static void queue_deinit(task_queue_t *q) {
    qemu_cond_init(&q->cv);
}

static int init_device_lib(EduState *vta, const char* name) {
    void* handle = dlopen("libvta_device.so", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        goto error;
    }
    vta->alloc = (device_alloc)dlsym(handle, "VTAMemAlloc");
    vta->free = (device_free)dlsym(handle, "VTADeviceFree");
    vta->virt_phys = (device_virt_to_phys)dlsym(handle, "VTAMemGetPhyAddr");
    vta->exec = (device_exec)dlsym(handle, "VTADeviceExec");

    if (!vta->alloc || !vta->free || !vta->virt_phys || !vta->exec)
        goto error;

    return 0;

error:
    fprintf(stderr, "error in finding the device function stub\n");
    dlclose(handle);
    return 1;
}

static bool vta_msi_enabled(EduState *vta)
{
    return msi_enabled(&vta->pdev);
}

static void vta_raise_irq(EduState *vta, uint32_t val)
{
    vta->irq_status |= val;
    if (vta->irq_status) {
        if (vta_msi_enabled(vta)) {
            msi_notify(&vta->pdev, 0);
        } else {
            pci_set_irq(&vta->pdev, 1);
        }
    }
}

static uint64_t vta_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    EduState *vta = opaque;
    uint64_t val = 0;
    if (addr < VTA_CTL_SIZE) {
        if (addr < sizeof(vta->main_ctl)) {
            int idx = addr / sizeof(vta->main_ctl[0]);
            int offset = (addr % sizeof(vta->main_ctl[0])) / sizeof(uint32_t);
            val = vta->main_ctl[idx].data[offset];
        }
        // fprintf(stderr, "read control msg %d\n", val);
    } else {
        fprintf(stderr, "out of bound read\n");
    }
    
    return val;
}

static void vta_mmio_write(void *opaque, hwaddr addr, uint64_t val,
                unsigned size)
{
    EduState *vta = opaque;

    if (addr < VTA_CTL_SIZE) {
        // fprintf(stderr, "receive control write msg\n");
        if (addr < sizeof(vta->main_ctl)) {
            int idx = addr / sizeof(vta->main_ctl[0]);
            int offset = (addr % sizeof(vta->main_ctl[0])) / sizeof(uint32_t);
            vta->main_ctl[idx].data[offset] = val;
            if (offset == VTA_CTRL_STATUS_OFFSET && val == 1) {
                queue_push(&vta->queue_ctl, idx);
            }
        }
    } else {
        fprintf(stderr, "out of bound write\n");
    }
}

static const MemoryRegionOps vta_mmio_ops = {
    .read = vta_mmio_read,
    .write = vta_mmio_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 8,
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 8,
    },

};

/*
 * We purposely use a thread, so that users are forced to wait for the status
 * register.
 */
static void *vta_fact_thread(void *opaque)
{
    EduState *vta = opaque;

    while (1) {

        int idx = queue_pop(&vta->queue_ctl, &vta->stopping);

        if (vta->stopping) break;

        fprintf(stderr, "executing %d task at %u, %u, %u, %u\n", idx, vta->main_ctl[idx].insn_phy_addr, vta->main_ctl[idx].offset, vta->main_ctl[idx].insn_count, vta->main_ctl[idx].wait_cycles);
        vta->exec(vta->main_ctl[idx].insn_phy_addr, vta->main_ctl[idx].offset, vta->main_ctl[idx].insn_count, vta->main_ctl[idx].wait_cycles);
        vta->main_ctl[idx].status = 0;

    }

    return NULL;
}

/*
static void* allocate_align(int align, int size) {
    unsigned long mask = align - 1;
    void* mem = malloc(size + mask);
    void* ptr = ((unsigned long)(mem + mask) & (~mask));
    return ptr;
}
*/

static void pci_vta_realize(PCIDevice *pdev, Error **errp)
{
    EduState *vta = VTA(pdev);
    uint8_t *pci_conf = pdev->config;

    pci_config_set_interrupt_pin(pci_conf, 1);

    if (msi_init(pdev, 0, 1, true, false, errp)) {
        return;
    }

    queue_init(&vta->queue_ctl);

    qemu_mutex_init(&vta->thr_mutex);
    qemu_cond_init(&vta->thr_cond);
    qemu_thread_create(&vta->thread, "vta", vta_fact_thread,
                       vta, QEMU_THREAD_JOINABLE);

    memory_region_init_io(&vta->mmio, OBJECT(vta), &vta_mmio_ops, vta,
                    "vta-mmio", VTA_CTL_SIZE);
    pci_register_bar(pdev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &vta->mmio);

    init_device_lib(vta, "libvta_device.so");
    vta->mem = vta->alloc(VTA_MEM_SIZE, 0);
    // fprintf(stderr, "allocate %lx\n", VTA_MEM_SIZE);

    memory_region_init_ram_device_ptr(&vta->ram, OBJECT(vta), "vta-ram", VTA_MEM_SIZE, vta->mem);
    // memory_region_init_io(&vta->ram, OBJECT(vta), &vta_ram_ops, vta,
                // "vta-ram", VTA_MEM_SIZE);
    pci_register_bar(pdev, 1, PCI_BASE_ADDRESS_SPACE_MEMORY
                            | PCI_BASE_ADDRESS_MEM_TYPE_64, &vta->ram);
}

static void pci_vta_uninit(PCIDevice *pdev)
{
    EduState *vta = VTA(pdev);

    qemu_mutex_lock(&vta->thr_mutex);
    vta->stopping = true;
    qemu_mutex_unlock(&vta->thr_mutex);
    qemu_cond_signal(&vta->thr_cond);
    qemu_thread_join(&vta->thread);

    qemu_cond_destroy(&vta->thr_cond);
    qemu_mutex_destroy(&vta->thr_mutex);

    queue_deinit(&vta->queue_ctl);

    msi_uninit(pdev);
}

static void vta_instance_init(Object *obj)
{
    EduState *vta = VTA(obj);

    vta->dma_mask = (1UL << 28) - 1;
    object_property_add_uint64_ptr(obj, "dma_mask",
                                   &vta->dma_mask, OBJ_PROP_FLAG_READWRITE);
    vta->ram_size_gb = 1;
    object_property_add_uint32_ptr(obj, "ram_size_gb",
                                   &vta->ram_size_gb, OBJ_PROP_FLAG_READWRITE);
}

static void vta_class_init(ObjectClass *class, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(class);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(class);

    k->realize = pci_vta_realize;
    k->exit = pci_vta_uninit;
    k->vendor_id = PCI_VENDOR_ID_QEMU;
    k->device_id = 0x11e9;
    k->revision = 0x10;
    k->class_id = PCI_CLASS_OTHERS;
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static void pci_vta_register_types(void)
{
    static InterfaceInfo interfaces[] = {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    };
    static const TypeInfo vta_info = {
        .name          = TYPE_PCI_VTA_DEVICE,
        .parent        = TYPE_PCI_DEVICE,
        .instance_size = sizeof(EduState),
        .instance_init = vta_instance_init,
        .class_init    = vta_class_init,
        .interfaces = interfaces,
    };

    type_register_static(&vta_info);
}
type_init(pci_vta_register_types)
