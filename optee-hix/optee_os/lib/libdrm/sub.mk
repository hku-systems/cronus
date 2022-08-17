
incdirs-y += .
incdirs-y += include
incdirs-y += include/drm
incdirs-y += libatomic_ops/src/

muslcdir := ./lib/libutils-user/muslc

MUSLC_ARCH := arm
ifeq ($(ta-target),ta_arm64)
MUSLC_ARCH := aarch64
endif

cflags-y += -I$(muslcdir)/arch/$(MUSLC_ARCH)/
cflags-y += -I$(muslcdir)/arch/generic/
# cflags-y += -I$(srcdir)/obj/src/internal
cflags-y += -I$(muslcdir)/src/include
cflags-y += -I$(muslcdir)/include
cflags-y += -Icore/include/linux_headers/linux_include/uapi
cflags-y += -Icore/include/linux_headers/arch_header/uapi
cflags-y += -Icore/include/linux_headers/arch_header/generated/uapi
cflags-y += -Icore/include/linux_headers/linux_include

cflags-y += -include lib/libdrm/include/cconfig.h
cflags-y += -DMAJOR_IN_SYSMACROS

drm-nouveau-y := nouveau/abi16.c nouveau/bufctx.c nouveau/nouveau.c	nouveau/pushbuf.c

srcs-y += xf86drm.c xf86drmHash.c xf86drmRandom.c xf86drmSL.c xf86drmMode.c 
srcs-y += $(drm-nouveau-y)