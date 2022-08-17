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


int tee_user_gettid() {
    return 111;
}

int tee_user_getpid() {
    return 1;
}