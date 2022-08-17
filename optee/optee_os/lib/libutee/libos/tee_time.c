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
#include "tee_syscalls.h"

#ifdef __aarch64__

static inline __attribute__((no_instrument_function))uint64_t
read_cntpct(void) {
  uint64_t val64 = 0;asm volatile("mrs %0, " "cntpct_el0":"=r"(val64));
  return val64;
}

static inline __attribute__((no_instrument_function))uint32_t
read_cntfrq(void) {
  uint64_t val64 = 0;asm volatile("mrs %0, " "cntfrq_el0":"=r"(val64));
  return val64;
}

static inline uint64_t barrier_read_cntpct(void)
{
  asm volatile ("isb");
  return read_cntpct();
}

#define TEE_TIME_MICRO_BASE    1000000

static void timespec_now(struct timespec *ts) {
	uint64_t cntpct = barrier_read_cntpct();
	uint32_t cntfrq = read_cntfrq();

	ts->tv_sec = cntpct / cntfrq;
	ts->tv_nsec = (cntpct % cntfrq) / (cntfrq / TEE_TIME_MICRO_BASE) * 1000;
}

int tee_clock_gettime(int clk, struct timespec *ts) {
  timespec_now(ts);
  libos_debug("get time %d:%d\n", ts->tv_sec, ts->tv_nsec);
	return 0;
}

int tee_nanosleep(const struct timespec *request,
                    struct timespec *remain) {
  struct timespec cur, start, deadline;
  timespec_now(&start);
  cur = start;
  
  deadline.tv_sec = start.tv_sec + request->tv_sec;
  deadline.tv_nsec = start.tv_nsec + request->tv_nsec;

  while (1) {
    timespec_now(&cur);
    if (cur.tv_sec > deadline.tv_sec 
      || (cur.tv_sec == deadline.tv_sec && cur.tv_nsec > deadline.tv_nsec)) {
        break;
      }
  }
  libos_debug("sleep for some time\n");
  return 0;  
}

int tee_clock_nanosleep(clockid_t clockid, int flags,
                           const struct timespec *request,
                           struct timespec *remain) {
  return tee_nanosleep(request, remain);
}

#else
int tee_clock_gettime(int clk, struct timespec *ts) {
	return -1;
}

int tee_clock_nanosleep(clockid_t clockid, int flags,
                           const struct timespec *request,
                           struct timespec *remain) {
  return 1;
}
int tee_nanosleep(const struct timespec *request,
                  struct timespec *remain) {
  return 1;
}
#endif