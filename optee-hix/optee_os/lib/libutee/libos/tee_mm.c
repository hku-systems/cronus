/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <tee_internal_api_extensions.h>
#include <utee_syscalls.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/uio.h>
#include <pthread.h>
#include <pthread_arch.h>

#include "libos.h"
#include "tee_syscalls.h"


void *tee_mmap(void *addr, size_t length, int prot, int flags,
                int fd, off_t offset) {
    void *mapped_addr = NULL;
    int sys_fd = fd_struct_mapping[fd].handle;
    TEE_Result ret = _utee_mmap(addr, length, prot, flags, sys_fd, offset, &mapped_addr);
    if (ret) {
        mapped_addr = (void*) -1;
        libos_err_ret2(mapped_addr, ret);
    }
    libos_debug_ret(mapped_addr);
}

void *tee_mremap(void *old_address, size_t old_size,
            size_t new_size, int flags) {
    void *new_addr;
    TEE_Result ret = _utee_mremap(old_address, old_size, new_size, flags, &new_addr);
    if (ret) {
        new_addr = (void*) -1;
        libos_err_ret2(new_addr, ret);
    }
    libos_debug_ret(new_addr);
}

int tee_munmap(void *addr, size_t len) {
    int ret_status;
    TEE_Result ret = _utee_munmap(addr, len, &ret_status);
    if (ret) {
        ret_status = -1;
        libos_err_ret2(ret_status, ret);
    }
    libos_debug_ret(ret_status);
}