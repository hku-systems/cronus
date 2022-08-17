/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <stdio.h>
#include <string.h>
#include <utee_syscalls.h>

#include <sys/uio.h>

#include "libos.h"
#include "tee_syscall_musl.h"
#include "tee_syscalls.h"

#define DISPATCH_SYSCALL_RET(id, cmd) case SYS_##id: ret = cmd; break
#define DISPATCH_SYSCALL_RET_0(id, cmd) case SYS_##id: cmd; break

void tee_syscall_none() {
    libos_err("debug none\n");
}

static long __syscall_n(long n, long a, long b, long c, long d, long e, long f) {
    long ret = 0;

    switch (n) {
        DISPATCH_SYSCALL_RET(mmap,    tee_mmap((void*)a, (size_t)b, (int)c, (int)d, (int)e, (off_t)f));
        DISPATCH_SYSCALL_RET(mremap,  tee_mremap((void*)a, (size_t)b, (size_t)c, (int)d));
        DISPATCH_SYSCALL_RET(munmap,  tee_munmap((void*)a, (size_t)b));
        DISPATCH_SYSCALL_RET(openat,  tee_fs_openat((int)a, (const char*)b, c, d));
        DISPATCH_SYSCALL_RET(read,    tee_fs_read((int)a, (void*)b, (int)c));
        DISPATCH_SYSCALL_RET(readv,   tee_fs_readv((int)a, (const struct iovec*)b, c));
        DISPATCH_SYSCALL_RET(writev,  tee_fs_writev((int)a, (const struct iovec*)b, c));
        DISPATCH_SYSCALL_RET(newfstatat,  tee_fs_newfstatat((int)a, (const char*)b, (struct stat*)c, (int)d));
        DISPATCH_SYSCALL_RET(fstat,  tee_fs_fstat((int)a, (struct stat*)b));
        DISPATCH_SYSCALL_RET(ioctl,   tee_dev_ioctl((int)a, (unsigned long)b, (void*)c));
        DISPATCH_SYSCALL_RET_0(close, tee_fs_close((int)a));
        DISPATCH_SYSCALL_RET(gettid, tee_user_gettid());
        DISPATCH_SYSCALL_RET(getpid, tee_user_getpid());
        DISPATCH_SYSCALL_RET(sched_yield, tee_sched_yield());
        DISPATCH_SYSCALL_RET(clock_gettime, tee_clock_gettime((int)a, (void*)b));
        DISPATCH_SYSCALL_RET(clock_nanosleep, tee_clock_nanosleep((int)a, (int)b, (void*)c, (void*)d));
        DISPATCH_SYSCALL_RET(nanosleep, tee_nanosleep((void*)a, (void*)b));
        default:
            ret = -1;
            libos_err("unimplemented syscall %ld (%lx %lx %lx %lx %lx %lx) -> %ld\n", n, a, b, c, d, e, f, ret);
            tee_syscall_none();
            break;
    }

    return ret;
}

long __syscall0(long n) {
    return __syscall_n(n, 0, 0, 0, 0, 0, 0);
}

long __syscall1(long n, long a) {
    return __syscall_n(n, a, 0, 0, 0, 0, 0); 
}

long __syscall2(long n, long a, long b) {
    return __syscall_n(n, a, b, 0, 0, 0, 0);
}

long __syscall3(long n, long a, long b, long c) {
    return __syscall_n(n, a, b, c, 0, 0, 0);
}

long __syscall4(long n, long a, long b, long c, long d) {
    return __syscall_n(n, a, b, c, d, 0, 0);
}

long __syscall5(long n, long a, long b, long c, long d, long e) {
    return __syscall_n(n, a, b, c, d, e, 0);
}

long __syscall6(long n, long a, long b, long c, long d, long e, long f) {
    return __syscall_n(n, a, b, c, d, e, f);
}