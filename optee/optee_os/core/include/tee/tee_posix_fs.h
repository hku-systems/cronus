/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#pragma once


#include <stdlib.h>
#include <string_ext.h>
#include <string.h>
#include <tee/tee_fs.h>
#include <trace.h>
#include <util.h>

struct posix_fs {
    TEE_Result (*open)(const char *filename, int flag, int mode, int *fd);
    TEE_Result (*read)(int fd, void *buf, size_t count, size_t off, ssize_t *rsize);
    TEE_Result (*write)(int fd, const void *buf, size_t count, size_t off, ssize_t *wsize);
    TEE_Result (*close)(int fd);
};

extern const struct posix_fs posix_direct_ree_fs;

#define POSIX_DEV_MINIMUM_FD 200

struct posix_dev {
    TEE_Result (*open)(const char *filename, int flag, int mode, int *fd);
    TEE_Result (*ioctl)(int fd, unsigned long cmd, void* arg, int *ret_val);
    TEE_Result (*close)(int fd);
};

extern const char* dev_prefix;
extern const struct posix_dev posix_device_ctl;