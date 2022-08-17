/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <asm/irq.h>

#include <linux/device.h>
#include <linux/fs.h>
#include <linux/i2c.h>
#include <linux/interrupt.h>
#include <linux/mount.h>
#include <linux/kdev_t.h>
#include <linux/kernel.h>
#include <linux/platform_device.h>
#include <linux/sched.h>
#include <linux/slab_def.h>
#include <linux/slab.h>

#include "tee-port.h"

static struct {
    unsigned int major;
    struct file_operations *fops;
} char_device_mapping[20];

static int char_device_cnt = 0;
static int max_major_id = 2;
int __register_chrdev(unsigned int major, unsigned int baseminor,
			     unsigned int count, const char *name,
			     const struct file_operations *fops) {
    int in_major = major;
    if (major == 0) {
        major = max_major_id + 1;
        max_major_id++;
    } else {
        if (max_major_id < major)
            max_major_id = major;
    }

    // FIXME: check conflicts
    char_device_mapping[char_device_cnt].major = major;
    char_device_mapping[char_device_cnt++].fops = fops;

    return in_major ? 0 : major;
}

void __unregister_chrdev(unsigned int major, unsigned int baseminor,
				unsigned int count, const char *name) NO_IMPL;

static struct file_operations * chrdev_get_fops(unsigned int major) {
    for (int i = 0;i < char_device_cnt;i++) {
        if (char_device_mapping[i].major == major)
            return char_device_mapping[i].fops;
    }
    return NULL;
}

// we are actually returning file instead of fop
void* posix_get_device_ops(const char *dev_name) {
    struct device *dev = devtmpfs_get_node(dev_name);
    if (!dev) {
        printk("cannot find device %s\n", dev_name);
        return NULL;
    }
    int major = MAJOR(dev->devt);
    struct file_operations *fop;
    struct file *f = (struct file*)malloc(sizeof(struct file));
    struct inode node;
    fop = chrdev_get_fops(major);
    if (!fop) {
        printk("error in finding chardev's fopes\n");
        return NULL;
    }
    node.i_rdev = dev->devt;
    f->f_op = fop;
    if (fop->open(&node, f)) {
        printk("error in driver's open\n");
        return NULL;
    }
    return f;
}

int posix_do_device_ioctl(void *ops, unsigned long cmd, void* arg) {
    struct file *f = (struct file*) ops;
    return f->f_op->unlocked_ioctl(f, cmd, (unsigned long)arg);
}