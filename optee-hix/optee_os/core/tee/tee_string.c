/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <asm/irq.h>

#include <linux/i2c.h>
#include <linux/interrupt.h>
#include <linux/kernel.h>
#include <linux/platform_device.h>
#include <linux/sched.h>
#include <linux/slab_def.h>
#include <linux/slab.h>

#include "tee-port.h"

void __warn_printk(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    trace_vprintf(__warn_printk, 0, 0, 0, fmt, ap);
    va_end(ap);
}

int printk(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    trace_vprintf(0, 0, 0, 0, fmt, ap);
    va_end(ap);
}

int __printk_ratelimit(const char *func) NO_IMPL;

extern void __do_panic(const char *file, const int line, const char *func,
		const char *msg);

void warn_slowpath_fmt(const char *file, const int line, unsigned taint,
                       const char *fmt, ...) {
    // __do_panic(file, line, "", fmt);
    printk("panic at %s:%d %s\n", file, line, fmt);
}

int vprintk(const char *s, va_list args) {
    trace_vprintf(0, 0, 0, 0, s, args);
    return 0;
}

int vprintk_emit(int facility, int level,
                 const char *dict, size_t dictlen,
                 const char *fmt, va_list args) {
    char* first = kvasprintf(0, fmt, args);
    va_arg(args, char*);
    va_arg(args, char*);
    struct va_format *vaf = va_arg(args, struct va_format *);

    int total_size = strlen(first) + strlen(vaf->fmt) + 2;
    char *buf = malloc(total_size);
    memset(buf, 0, total_size);
    strcpy(buf, first);
    buf[strlen(first)] = ' ';
    buf[strlen(first) + 1] = 0;
    strcat(buf, vaf->fmt);
    trace_vprintf(0, 0, 0, 0, buf, *(vaf->va));
    free(first);
}

int sscanf(const char * buf, const char * format, ...) {
    return 0;
}

int scnprintf(char *buf, size_t size, const char *fmt, ...) {
    unimplemented();
}


long simple_strtol(const char *cp, char **endp, unsigned int base) NO_IMPL_0;

void seq_printf(struct seq_file *m, const char *f, ...) {
    printk(f);
}