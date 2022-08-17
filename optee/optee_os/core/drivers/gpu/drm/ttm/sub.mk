# SPDX-License-Identifier: GPL-2.0
#
# Makefile for the drm device driver.  This driver provides support for the

ttm-y := ttm_memory.c ttm_tt.c ttm_bo.c \
	ttm_bo_util.c ttm_bo_vm.c ttm_module.c \
	ttm_execbuf_util.c ttm_page_alloc.c ttm_bo_manager.c
ttm-$(CONFIG_AGP) += ttm_agp_backend.c
ttm-$(CONFIG_DRM_TTM_DMA_PAGE_POOL) += ttm_page_alloc_dma.c

srcs-$(CONFIG_DRM_TTM) += $(ttm-y)
