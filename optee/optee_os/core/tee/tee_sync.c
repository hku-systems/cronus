/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <linux/kernel.h>
#include <linux/platform_device.h>
#include <linux/sched.h>
#include <linux/slab_def.h>
#include <linux/slab.h>

#include "tee-port.h"

void __mutex_init(struct mutex *lock, const char *name, struct lock_class_key *key)
{
    struct wait_queue wq = { .slh_first = NULL };
    lock->wq = wq;
    lock->spin_lock  = 0;
    lock->state = 0;
}

int mutex_lock_killable(struct mutex *lock) {
    mutex_lock(lock);
    return 0;
}

void __init_rwsem(struct rw_semaphore *sem, const char *name,
			 struct lock_class_key *key) {
    unimplemented();             
}

int /* __must_check */ ww_mutex_lock(struct ww_mutex *lock, struct ww_acquire_ctx *ctx) {
    // TODO
    return 0;
}

int ww_mutex_lock_interruptible(struct ww_mutex *lock,
						    struct ww_acquire_ctx *ctx) {
    // TODO
    return 0;
}

void ww_mutex_unlock(struct ww_mutex *lock) {
    // TODO
}

void down_read(struct rw_semaphore *sem) {
    unimplemented();
}

void down_write(struct rw_semaphore *sem) {
    unimplemented();
}

int down_write_killable(struct rw_semaphore *sem) NO_IMPL_0;

void up_read(struct rw_semaphore *sem) {
    unimplemented();
}

void up_write(struct rw_semaphore *sem) {
    unimplemented();
}

void console_lock(void) {
    unimplemented();
}

void console_unlock(void) {
    unimplemented();
}

bool mutex_is_locked(struct mutex *lock) {
    // unimplemented();
    return lock->state != 0;
}

int mutex_lock_interruptible(struct mutex *lock) {
    // unimplemented();
    mutex_lock(lock);
    return 0;
}

void
prepare_to_wait_exclusive(struct wait_queue_head *wq_head,
        struct wait_queue_entry *wq_entry, int state) NO_IMPL;