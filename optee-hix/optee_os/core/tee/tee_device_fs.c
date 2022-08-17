/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <assert.h>
#include <kernel/tee_misc.h>
#include <kernel/thread.h>
#include <mm/core_memprot.h>
#include <optee_rpc_cmd.h>
#include <stdlib.h>
#include <string_ext.h>
#include <string.h>
#include <tee/tee_posix_fs.h>
#include <tee/tee_fs.h>
#include <tee/tee_fs_rpc.h>
#include <tee/tee_pobj.h>
#include <tee/tee_svc_storage.h>
#include <trace.h>
#include <unistd.h>
#include <util.h>

struct device;

const char* dev_prefix = "/dev/";

#define MAXIMUM_DEV_MAPPING 20

struct {
    void *fops;
} dev_mapping[MAXIMUM_DEV_MAPPING];

static int dev_mapping_cnt = 0;

extern void* posix_get_device_ops(const char *dev_name);
extern int posix_do_device_ioctl(void *ops, unsigned long cmd, void* arg);

TEE_Result dev_open(const char *filename, int flag, int mode, int *fd) {
    const char* dev_name = filename + strlen(dev_prefix);
    void *fops = posix_get_device_ops(dev_name);
    int ret_fd = 0;
    if (!fops) {
        EMSG("errnor in finding device operation %s\n", dev_name);
        return TEE_ERROR_BAD_PARAMETERS;
    }

    ret_fd = dev_mapping_cnt++;
    dev_mapping[ret_fd].fops = fops;
    *fd = ret_fd + POSIX_DEV_MINIMUM_FD;
    return TEE_SUCCESS;
}

TEE_Result dev_ioctl(int fd, unsigned long cmd, void* arg, int *ret_val) {
    if (fd - POSIX_DEV_MINIMUM_FD > MAXIMUM_DEV_MAPPING) {
        EMSG("fd out of range %d", fd);
        return TEE_ERROR_BAD_PARAMETERS;
    }
    void* fops = dev_mapping[fd - POSIX_DEV_MINIMUM_FD].fops;
    if (!fops) {
        EMSG("cannot find dev fops");
        return TEE_ERROR_BAD_PARAMETERS;
    }
    *ret_val = posix_do_device_ioctl(fops, cmd, arg);
    if (*ret_val) {
        EMSG("doing ioctl error with %lx", ret_val);
        return TEE_ERROR_BAD_PARAMETERS; 
    }
    return TEE_SUCCESS;
}

TEE_Result dev_close(int fd) {
    dev_mapping[fd].fops = NULL;
    return TEE_SUCCESS;
}

const struct posix_dev posix_device_ctl = {
    .open = dev_open,
    .ioctl = dev_ioctl,
    .close = dev_close
};