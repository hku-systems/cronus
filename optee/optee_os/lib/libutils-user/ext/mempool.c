// SPDX-License-Identifier: BSD-2-Clause
/*
 * Copyright (c) 2014, STMicroelectronics International N.V.
 * Copyright (c) 2018-2019, Linaro Limited
 */


#include <assert.h>
#include <compiler.h>
#include <malloc.h>
#include <mempool.h>
#include <string.h>
#include <util.h>

#if defined(__KERNEL__)
#include <kernel/mutex.h>
#include <kernel/panic.h>
#include <kernel/thread.h>
#endif

/*
 * Allocation of temporary memory buffers which are used in a stack like
 * fashion. One exmaple is when a Big Number is needed for a temporary
 * variable in a Big Number computation: Big Number operations (add,...),
 * crypto algorithms (rsa, ecc,,...).
 *
 *  The allocation algorithm takes memory buffers from a pool,
 *  characterized by (cf. struct mempool):
 * - the total size (in bytes) of the pool
 * - the offset of the last item allocated in the pool (struct
 *   mempool_item). This offset is -1 is nothing is allocated yet.
 *
 * Each item consists of (struct mempool_item)
 * - the size of the item
 * - the offsets, in the pool, of the previous and next items
 *
 * The allocation allocates an item for a given size.
 * The allocation is performed in the pool after the last
 * allocated items. This means:
 * - the heap is never used.
 * - there is no assumption on the size of the allocated memory buffers. Only
 *   the size of the pool will limit the allocation.
 * - a constant time allocation and free as there is no list scan
 * - but a potentially fragmented memory as the allocation does not take into
 *   account "holes" in the pool (allocation is performed after the last
 *   allocated variable). Indeed, this interface is supposed to be used
 *   with stack like allocations to avoid this issue. This means that
 *   allocated items:
 *   - should have a short life cycle
 *   - if an item A is allocated before another item B, then A should be
 *     released after B.
 *   So the potential fragmentation is mitigated.
 */


struct mempool {
	size_t size;  /* size of the memory pool, in bytes */
	ssize_t last_offset;   /* offset to the last one */
	vaddr_t data;
#ifdef CFG_MEMPOOL_REPORT_LAST_OFFSET
	ssize_t max_last_offset;
#endif
#if defined(__KERNEL__)
	void (*release_mem)(void *ptr, size_t size);
	struct recursive_mutex mu;
#endif
};


// TODO: dummy impl

struct mempool *
mempool_alloc_pool(void *data, size_t size,
		   void (*release_mem)(void *ptr, size_t size) __maybe_unused)
{

}

void *mempool_alloc(struct mempool *pool, size_t size)
{

}

void *mempool_calloc(struct mempool *pool, size_t nmemb, size_t size)
{

}

void mempool_free(struct mempool *pool, void *ptr)
{

}
