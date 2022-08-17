/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <tee_internal_api_extensions.h>
#include <utee_syscalls.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/uio.h>

#include "libos.h"

long dev_ioctl(int fd, unsigned long request, void* args) {
    struct fd_struct *fds = &fd_struct_mapping[fd];
    TEE_Result ret = _utee_dev_ioctl(fd, request, args);
    if (ret) {
        return -1;
    }
    libos_debug_ret(ret);
}