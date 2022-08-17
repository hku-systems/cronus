/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <stdlib.h>
#include <string.h>

#include <kernel/interrupt.h>


typedef int (*irq_handler_t)(int, void *);

#define IRQ_CB_MAX_NAME 20

static struct irq_callback {
    int irq;
    int name[IRQ_CB_MAX_NAME];
    irq_handler_t handler;
    irq_handler_t thread_fn;
    void* data;
};

static enum itr_return itr_cb(struct itr_handler *h)
{
	struct irq_callback *cs = (struct irq_callback*)h->data;
    irq_handler_t handler = cs->handler;
    irq_handler_t thread_fn = cs->thread_fn;
#ifdef DEBUG_CRONUS    
    printk("handling irq %d (%s) -> (%lx, %lx) (%lx)\n", cs->irq, cs->name, handler, thread_fn, cs->data);
#endif
    handler(cs->irq, cs->data);
    if (thread_fn) {
        handler(cs->irq, cs->data);
#ifdef DEBUG_CRONUS
        printk("handling irq %d bottom half fin\n");
#endif
    }
	return 1;
}

static int sample_cb(int irq, void *data) {
    printk("irq %d is called\n", irq);
    printk("irq %d is called\n", irq);
    return 0;
}

int request_irq_internal(int irq, irq_handler_t handler, irq_handler_t thread_fn, const char *name, void *dev) {
    struct itr_handler *wdt_itr;
    struct irq_callback *cs = (struct irq_callback *) malloc(sizeof(struct irq_callback));
    memset(cs, 0, sizeof(struct irq_callback));
    irq += 32; // for external irq
    
    cs->irq = irq;
    cs->handler = handler;
    cs->thread_fn = thread_fn;
    cs->data = dev;
    memcpy(cs->name, name, (strlen(name) < IRQ_CB_MAX_NAME) ? strlen(name) : IRQ_CB_MAX_NAME - 1);

    printk("irq %d is registered to (%lx %lx)\n", irq, handler, thread_fn);
	wdt_itr = itr_alloc_add(irq, itr_cb,
				1 << 0, cs);
	if (!wdt_itr) {
        printk("out of memory in irq\n");
		return 1;
    }

	itr_enable(wdt_itr->it);
    return 0;
}