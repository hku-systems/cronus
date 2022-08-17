# SPDX-License-Identifier: GPL-2.0
srcs-y					+= heap-helpers.c
srcs-$(CONFIG_DMABUF_HEAPS_SYSTEM)	+= system_heap.c
srcs-$(CONFIG_DMABUF_HEAPS_CMA)		+= cma_heap.c
