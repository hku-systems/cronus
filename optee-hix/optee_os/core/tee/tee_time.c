/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <linux/delay.h>
#include <linux/kernel.h>
#include <linux/platform_device.h>
#include <linux/sched.h>
#include <linux/slab_def.h>
#include <linux/slab.h>

#include "tee-port.h"
#define TEE_TIME_MILLIS_BASE    1000
#define TEE_TIME_MICROS_BASE    1000000

unsigned long clk_get_rate(struct clk *clk) {
    unimplemented();
}

void add_timer(struct timer_list *timer) NO_IMPL;

void add_timer_on(struct timer_list *timer, int cpu) NO_IMPL;

int del_timer(struct timer_list * timer) NO_IMPL;

int mod_timer(struct timer_list *timer, unsigned long expires) NO_IMPL;

ktime_t ktime_get(void) {
    uint64_t cntpct = barrier_read_cntpct();
    uint32_t cntfrq = read_cntfrq();
    uint64_t cur_usecs = cntpct / (cntfrq / TEE_TIME_MICROS_BASE);
    return (ktime_t)cur_usecs;
}

static void do_init_timer(struct timer_list *timer,
                          void (*func)(struct timer_list *),
                          unsigned int flags,
                          const char *name, struct lock_class_key *key)
{
    timer->entry.pprev = NULL;
    timer->function = func;
    timer->flags = flags | raw_smp_processor_id();
    lockdep_init_map(&timer->lockdep_map, name, key, 0);
}

void init_timer_key(struct timer_list *timer,
                    void (*func)(struct timer_list *), unsigned int flags,
                    const char *name, struct lock_class_key *key) {
    do_init_timer(timer, func, flags, name, key);
}

//unsigned long volatile jiffies;

u8 jiffies_read_cnt = 0;
u64 jiffies_value = 0;

unsigned int jiffies_to_msecs(const unsigned long j) {
  uint32_t cntfrq = read_cntfrq();
  uint64_t cur_usecs = (j << CNTPCT_JIFFIES_SHIFT) / (cntfrq / TEE_TIME_MILLIS_BASE);
  return cur_usecs;
}

unsigned int jiffies_to_usecs(const unsigned long j) {
  uint32_t cntfrq = read_cntfrq();
  uint64_t cur_usecs = (j << CNTPCT_JIFFIES_SHIFT) / (cntfrq / TEE_TIME_MICROS_BASE);
  return cur_usecs;
}

unsigned long __msecs_to_jiffies(const unsigned int m) {
  uint32_t cntfrq = read_cntfrq();
  uint64_t cur_jiffies = m * (cntfrq / TEE_TIME_MILLIS_BASE);
  return cur_jiffies >> CNTPCT_JIFFIES_SHIFT;
}

u64 nsecs_to_jiffies64(u64 n) {
  return n;
}

void __ndelay(unsigned long nsecs) {
    uint64_t cur = nsecs = barrier_read_cntpct();
    while (barrier_read_cntpct() - cur < nsecs);
}

// void __const_udelay(unsigned long xloops) NO_IMPL;

unsigned long __usecs_to_jiffies(const unsigned int u) {
  uint32_t cntfrq = read_cntfrq();
  uint64_t jiff = u * (cntfrq / TEE_TIME_MICROS_BASE);   
  return jiff;
}

void __udelay(unsigned long usecs) {
    uint64_t cntpct = barrier_read_cntpct();
    uint32_t cntfrq = read_cntfrq();
    uint64_t cur_usecs = cntpct / (cntfrq / TEE_TIME_MICROS_BASE);

    while ((barrier_read_cntpct() / (cntfrq / TEE_TIME_MICROS_BASE)) - cur_usecs < usecs);
}

struct timespec64 ns_to_timespec64(const s64 nsec) NO_IMPL;

void msleep(unsigned int msecs) {
    uint64_t cntpct = barrier_read_cntpct();
    uint32_t cntfrq = read_cntfrq();
    uint64_t cur_msecs = cntpct / (cntfrq / TEE_TIME_MILLIS_BASE);

    while ((barrier_read_cntpct() / (cntfrq / TEE_TIME_MILLIS_BASE)) - cur_msecs < msecs);
}

void usleep_range(unsigned long min, unsigned long max) NO_IMPL;

extern uint32_t init_tee_time() {
  jiffies_value = jiffies_from_cntpct;
  return 0;
}