/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <linux/kthread.h>
#include <linux/slab.h>

#include "tee-port.h"

void *kthread_data(struct task_struct *task) NO_IMPL_NULL;

void kthread_bind_mask(struct task_struct *k, const struct cpumask *mask) {
    unimplemented();
}

struct task_struct *kthread_create_on_node(int (*threadfn)(void *data),
					   void *data,
					   int node,
					   const char namefmt[], ...) {
    unimplemented();						   
}

struct kthread_worker *
kthread_create_worker(unsigned int flags, const char namefmt[], ...) {
    unimplemented();
}

void kthread_destroy_worker(struct kthread_worker *worker) {
    unimplemented();
}

void kthread_flush_worker(struct kthread_worker *worker) {
    unimplemented();    
}

bool kthread_queue_work(struct kthread_worker *worker,
			struct kthread_work *work) {
    unimplemented();
}

bool kthread_should_stop(void) {
    unimplemented();
}

int kthread_stop(struct task_struct *k) {
    unimplemented();
}

static long
wait_for_common(struct completion *x, long timeout, int state)
{
    uint64_t last = jiffies;
    uint64_t cur = last;
    int retry = 0;
	do {
        schedule_yield();
        cur = jiffies;
        if (cur - last >= timeout) {
            printk("completion (x->done=%d) timeout with %d tries (cur=%lx, last=%lx, timeout=%lx)\n", x->done, retry, cur, last, timeout);
            return 0;
        }
        retry ++;
    } while (!x->done);
    return last + timeout - cur;
}

void wait_for_completion(struct completion *complete) {
    unimplemented();
}

int wait_for_completion_interruptible(struct completion *x) {
    unimplemented();
}

long wait_for_completion_interruptible_timeout(
	struct completion *x, unsigned long timeout) {
    unimplemented();
}

unsigned long wait_for_completion_timeout(struct completion *x,
						   unsigned long timeout) {
    return wait_for_common(x, timeout, TASK_UNINTERRUPTIBLE);
}

bool try_wait_for_completion(struct completion *x) {
    unimplemented();
}

long prepare_to_wait_event(struct wait_queue_head *wq_head,
                           struct wait_queue_entry *wq_entry, int state) NO_IMPL;

void schedule(void) NO_IMPL;

void finish_wait(struct wait_queue_head *wq_head, struct wait_queue_entry *wq_entry) NO_IMPL;

void init_wait_entry(struct wait_queue_entry *wq_entry, int flags) NO_IMPL;

int wake_up_state(struct task_struct *tsk, unsigned int state) NO_IMPL;
int wake_up_process(struct task_struct *tsk) NO_IMPL_0;
void __wake_up(struct wait_queue_head *wq_head, unsigned int mode, int nr, void *key) NO_IMPL;
void __wake_up_locked_key(struct wait_queue_head *wq_head, unsigned int mode, void *key) NO_IMPL;

void __set_task_comm(struct task_struct *tsk, const char *from, bool exec) NO_IMPL;

void sched_set_fifo(struct task_struct *p) NO_IMPL;

void swake_up_all_locked(struct swait_queue_head *q) NO_IMPL;
void swake_up_locked(struct swait_queue_head *q) NO_IMPL;

void complete(struct completion *x) {
    unsigned long flags;

	raw_spin_lock_irqsave(&x->wait.lock, flags);

	if (x->done != UINT_MAX)
		x->done++;
	swake_up_locked(&x->wait);
	raw_spin_unlock_irqrestore(&x->wait.lock, flags);
}

void complete_all(struct completion *x) {
    unsigned long flags;

	lockdep_assert_RT_in_threaded_ctx();

	raw_spin_lock_irqsave(&x->wait.lock, flags);
	x->done = UINT_MAX;
	swake_up_all_locked(&x->wait);
	raw_spin_unlock_irqrestore(&x->wait.lock, flags);
}

int _cond_resched(void) NO_IMPL_0;

int schedule_hrtimeout(ktime_t *expires, const enum hrtimer_mode mode) NO_IMPL_0;
long schedule_timeout(long timeout) NO_IMPL_0;
long schedule_timeout_interruptible(long timeout) NO_IMPL_0;

void __init_swait_queue_head(struct swait_queue_head *q, const char *name,
                             struct lock_class_key *key) {
    raw_spin_lock_init(&q->lock);
    lockdep_set_class_and_name(&q->lock, key, name);
    INIT_LIST_HEAD(&q->task_list);
}

void __init_waitqueue_head(struct wait_queue_head *wq_head, const char *name, struct lock_class_key *key) {
    spin_lock_init(&wq_head->lock);
    lockdep_set_class_and_name(&wq_head->lock, key, name);
    INIT_LIST_HEAD(&wq_head->head);
}
void __local_bh_enable_ip(unsigned long ip, unsigned int cnt) NO_IMPL;
void add_wait_queue(struct wait_queue_head *wq_head, struct wait_queue_entry *wq_entry) NO_IMPL;
void remove_wait_queue(struct wait_queue_head *wq_head, struct wait_queue_entry *wq_entry) NO_IMPL;
int default_wake_function(wait_queue_entry_t *curr, unsigned mode, int wake_flags,
			  void *key) NO_IMPL_0;

pid_t __task_pid_nr_ns(struct task_struct *task, enum pid_type type, struct pid_namespace *ns) NO_IMPL_0;

char *__get_task_comm(char *to, size_t len, struct task_struct *tsk) NO_IMPL_0;