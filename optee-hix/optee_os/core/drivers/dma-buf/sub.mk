# SPDX-License-Identifier: GPL-2.0-only
srcs-y := dma-buf.c dma-fence.c dma-fence-array.c dma-fence-chain.c \
	 dma-resv.c seqno-fence.c
srcs-$(CONFIG_DMABUF_HEAPS)	+= dma-heap.c
subdirs-$(CONFIG_DMABUF_HEAPS)	+= heaps/
srcs-$(CONFIG_SYNC_FILE)		+= sync_file.c
srcs-$(CONFIG_SW_SYNC)		+= sw_sync.c sync_debug.c
srcs-$(CONFIG_UDMABUF)		+= udmabuf.c

dmabuf_selftests-y := \
	selftest.c \
	st-dma-fence.c \
	st-dma-fence-chain.c

srcs-$(CONFIG_DMABUF_SELFTESTS)	+= $(dmabuf_selftests-y)
