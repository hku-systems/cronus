# SPDX-License-Identifier: GPL-2.0

srcs-$(CONFIG_HAS_DMA)			+= mapping.c direct.c
srcs-$(CONFIG_DMA_OPS)			+= dummy.c
srcs-$(CONFIG_DMA_CMA)			+= contiguous.c
srcs-$(CONFIG_DMA_DECLARE_COHERENT)	+= coherent.c
srcs-$(CONFIG_DMA_VIRT_OPS)		+= virt.c
srcs-$(CONFIG_DMA_API_DEBUG)		+= debug.c
srcs-$(CONFIG_SWIOTLB)			+= swiotlb.c
srcs-$(CONFIG_DMA_COHERENT_POOL)		+= pool.c
srcs-$(CONFIG_DMA_REMAP)			+= remap.c
