NOUVEAU_PATH ?= $(srctree)

# SPDX-License-Identifier: MIT
ccflags-y += -I $(NOUVEAU_PATH)/$(src)/include
ccflags-y += -I $(NOUVEAU_PATH)/$(src)/include/nvkm
ccflags-y += -I $(NOUVEAU_PATH)/$(src)/nvkm
ccflags-y += -I $(NOUVEAU_PATH)/$(src)

# NVKM - HW resource manager
#- code also used by various userspace tools/tests
subdirs-y += nvif

# NVIF - NVKM interface library (NVKM user interface also defined here)
#- code also used by various userspace tools/tests
subdirs-y += nvkm

# DRM - general
ifdef CONFIG_X86
nouveau-$(CONFIG_ACPI) += nouveau_acpi.c
endif
nouveau-$(CONFIG_DEBUG_FS) += nouveau_debugfs.c
nouveau-y += nouveau_drm.c
nouveau-y += nouveau_hwmon.c
nouveau-$(CONFIG_COMPAT) += nouveau_ioc32.c
nouveau-$(CONFIG_LEDS_CLASS) += nouveau_led.c
nouveau-y += nouveau_nvif.c
nouveau-$(CONFIG_NOUVEAU_PLATFORM_DRIVER) += nouveau_platform.c
nouveau-y += nouveau_usif.c # userspace <-> nvif
nouveau-y += nouveau_vga.c

# DRM - memory management
nouveau-y += nouveau_bo.c
nouveau-y += nouveau_bo0039.c
nouveau-y += nouveau_bo5039.c
nouveau-y += nouveau_bo74c1.c
nouveau-y += nouveau_bo85b5.c
nouveau-y += nouveau_bo9039.c
nouveau-y += nouveau_bo90b5.c
nouveau-y += nouveau_boa0b5.c
nouveau-y += nouveau_gem.c
nouveau-$(CONFIG_DRM_NOUVEAU_SVM) += nouveau_svm.c
nouveau-$(CONFIG_DRM_NOUVEAU_SVM) += nouveau_dmem.c
nouveau-y += nouveau_mem.c
nouveau-y += nouveau_prime.c
nouveau-y += nouveau_sgdma.c
nouveau-y += nouveau_ttm.c
nouveau-y += nouveau_vmm.c

# DRM - modesetting
nouveau-$(CONFIG_DRM_NOUVEAU_BACKLIGHT) += nouveau_backlight.c
nouveau-y += nouveau_bios.c
nouveau-y += nouveau_connector.c
nouveau-y += nouveau_display.c
nouveau-y += nouveau_dp.c
nouveau-y += nouveau_fbcon.c
nouveau-y += nv04_fbcon.c
nouveau-y += nv50_fbcon.c
nouveau-y += nvc0_fbcon.c

subdirs-y += dispnv04
subdirs-y += dispnv50

# DRM - command submission
nouveau-y += nouveau_abi16.c
nouveau-y += nouveau_chan.c
nouveau-y += nouveau_dma.c
nouveau-y += nouveau_fence.c
nouveau-y += nv04_fence.c
nouveau-y += nv10_fence.c
nouveau-y += nv17_fence.c
nouveau-y += nv50_fence.c
nouveau-y += nv84_fence.c
nouveau-y += nvc0_fence.c

srcs-$(CONFIG_DRM_NOUVEAU) += $(nouveau-y)
