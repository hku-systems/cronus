/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <linux/device.h>
#include <linux/types.h>
#include "base.h"

#define MAX_DEVNAME_SZ 20 

struct {
    char name[MAX_DEVNAME_SZ];
    struct device *dev;
} devtmpfs_list[100];

static int devtmpfs_list_cnt = 0;

int devtmpfs_create_node(struct device *dev) {
    umode_t mode;	/* 0 => delete */
	kuid_t uid;
	kgid_t gid;
    const char *tmp = NULL;
    device_get_devnode(dev, &mode, &uid, &gid, &tmp);
    if (tmp) {
        strcpy(devtmpfs_list[devtmpfs_list_cnt].name, tmp);
        printk("add dev %s to /dev/\n");
    }
    else
        printk("add dev (null) to /dev/\n");
    devtmpfs_list[devtmpfs_list_cnt++].dev = dev;
    return 0;
}



struct device* devtmpfs_get_node(const char *name) {
    for (int i = 0;i < devtmpfs_list_cnt;i++) {
        if (strcmp(name, devtmpfs_list[i].name) == 0) {
            return devtmpfs_list[i].dev;
        }
    }
    return NULL;
}

int devtmpfs_delete_node(struct device *dev) {
    // imp
}