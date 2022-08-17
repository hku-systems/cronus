/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#pragma once

#include <utee_syscalls.h>

extern char libos_log_buf[];
extern int libos_log_cnt;

#ifdef LIBOS_DEBUG
#define libos_debug(...) \
    libos_log_cnt = sprintf(libos_log_buf, "D/LIBOS:%s:%d ", __FUNCTION__, __LINE__); \
    sprintf(libos_log_buf + libos_log_cnt, __VA_ARGS__); \
    _utee_log(libos_log_buf, strlen(libos_log_buf));

#define libos_debug_ret(ret) \
    libos_debug("-> %lx\n", ret); \
    return ret;
#else
#define libos_debug(...)
#define libos_debug_ret(ret) \
    return ret;
#endif

#define libos_err(...) \
    libos_log_cnt = sprintf(libos_log_buf, "E/LIBOS:%s:%d ", __FUNCTION__, __LINE__); \
    sprintf(libos_log_buf + libos_log_cnt, __VA_ARGS__); \
    _utee_log(libos_log_buf, strlen(libos_log_buf));

#define libos_err_ret(ret) \
    libos_err("-> %lx\n", ret); \
    return ret;

#define libos_err_ret2(ret, retval) \
    libos_err("-> %lx (%lx)\n", ret, retval); \
    return ret;