/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <asm/irq.h>

#include <linux/device.h>
#include <linux/fs.h>
#include <linux/i2c.h>
#include <linux/interrupt.h>
#include <linux/mm.h>
#include <linux/mman.h>
#include <linux/mount.h>
#include <linux/kdev_t.h>
#include <linux/kernel.h>
#include <linux/platform_device.h>
#include <linux/sched.h>
#include <linux/slab_def.h>
#include <linux/slab.h>

extern struct {
    void *fops;
} dev_mapping[];

#define POSIX_DEV_MINIMUM_FD 200

#define PROTFLAG_TO_VMFLAG(prot, vm, flag) if (prot & PROT_##flag) vm |= VM_##flag

extern void core_mmu_get_user_va_range(long *base, size_t *size);

static long mmap_fd_base = 0;

void* mmap_fd(void *addr, size_t length, int prot, int flag,
				int fd, long offset) {

    struct file* f = dev_mapping[fd - POSIX_DEV_MINIMUM_FD].fops;
    struct vm_area_struct *vma;
    void* ret_addr = NULL;
    int vm_flags;
    int pkey = 0;

    int ret = 0;
    size_t ta_size;
    int pg_size = 0;

    // 0. calculate flag
    PROTFLAG_TO_VMFLAG(prot, vm_flags, READ);
    PROTFLAG_TO_VMFLAG(prot, vm_flags, WRITE);
    PROTFLAG_TO_VMFLAG(prot, vm_flags, EXEC);
    vm_flags |= (VM_MAYREAD | VM_MAYWRITE | VM_MAYEXEC);

    // 1. assign vaddr to vms
    // TODO: not a global value
    if (!mmap_fd_base) {
        core_mmu_get_user_va_range(&mmap_fd_base, &ta_size);
        mmap_fd_base += (16 << 22);
    }

    addr = mmap_fd_base;
    pg_size = (length + 4095) / 4096;
    mmap_fd_base += pg_size * 4096;

    vma = kmalloc(sizeof(struct vm_area_struct), __GFP_ZERO);
    vma->vm_file = f;
    vma->vm_start = addr;
    vma->vm_end = addr + length;
    vma->vm_flags = vm_flags;
    vma->vm_pgoff = (offset >> PAGE_SHIFT);

    // 2. do the mapping
    ret = f->f_op->mmap(f, vma);
    if (ret) {
        printk("mmap failed with %d\n", ret);
        goto err;
    }

    // 3. fault the addr
    // see memory.c:4416
    struct vm_fault *vmf = kmalloc(sizeof(struct vm_fault), __GFP_ZERO);
    vmf->vma = vma;
    vmf->address = addr;
    vmf->flags = vm_flags;
    vma->vm_ops->open(vma);
    ret = vma->vm_ops->fault(vmf);

    if (ret & VM_FAULT_ERROR) {
        printk("page fault handing error with %d\n", ret & VM_FAULT_ERROR);
        goto err;
    }

fin:
    return addr;
err:
    return NULL;
}

void* munmap_fd(void *addr, size_t length) {
    // unimplemented
    return NULL;
}


