/*
 * Copyright (c) 2016, Linaro Limited
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef BENCHMARK_AUX_H
#define BENCHMARK_AUX_H

#include <tee_client_api.h>
#include <tee_bench.h>

/* tee error code checking etc */
void tee_errx(const char *msg, TEEC_Result res);
void tee_check_res(TEEC_Result res, const char *errmsg);

/* misc aux print functions */
const char *bench_str_src(uint64_t source);
void print_line(void);

/* argv alloc/dealloc */
void alloc_argv(int argc, char *argv[], char **new_argv[]);
void dealloc_argv(int new_argc, char **new_argv);

void *mmap_paddr(intptr_t paddr, uint64_t size);
size_t get_library_load_offset(pid_t pid, const char *libname);

/* get amount of cores */
uint32_t get_cores(void);
#endif /* BENCHMARK_AUX_H */
