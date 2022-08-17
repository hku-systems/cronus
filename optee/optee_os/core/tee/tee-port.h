/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#pragma once

#include <linux/types.h>

#include <stdarg.h>

#define unimplemented() trace_printf(NULL, 0, 0, 0, "%s unimplemented\n", __FUNCTION__)
#define unstable() trace_printf(NULL, 0, 0, 0, "%s has an unstable implementation\n", __FUNCTION__)
#define NO_IMPL         { unimplemented(); }
#define NO_IMPL_0       { unimplemented(); return 0; }
#define NO_IMPL_NULL    { unimplemented(); return NULL; }
#define IMPL_NULL       { return NULL; }
#define IMPL_0          { return 0; }
#define IMPL_RET        {}

extern void *malloc(size_t size);
extern void free(void *ptr);
extern void *realloc(void *ptr, size_t size);

extern void trace_printf(const char *func, int line, int level, bool level_ok,
		  const char *fmt, ...) __printf(5, 6);

extern void trace_vprintf(const char *func, int line, int level, bool level_ok,
		   const char *fmt, va_list args) __printf(5, 0);

extern size_t __get_core_pos(void);

int schedule_yield(void);
void schedule_queue_info(void);