

muslcdir := ./lib/libutils-user/muslc

MUSLC_ARCH := arm
ifeq ($(ta-target),ta_arm64)
MUSLC_ARCH := aarch64
endif

# musl headers
cflags-y += -I$(muslcdir)/arch/$(MUSLC_ARCH)/
cflags-y += -I$(muslcdir)/arch/generic/
# cflags-y += -I$(srcdir)/obj/src/internal
cflags-y += -I$(muslcdir)/src/include
cflags-y += -I$(muslcdir)/include

# drm headers
drmdir := ./lib/libdrm/
cflags-y += -I$(drmdir)
cflags-y += -I$(drmdir)/nouveau
cflags-y += -I$(drmdir)/include/drm

# gdev headers
incdirs-y += gen
incdirs-y += common
incdirs-y += util
incdirs-y += lib/user
incdirs-y += lib/user/gdev
incdirs-y += lib/user/nouveau

# kernel headers
cflags-y += -Icore/include/linux_headers/linux_include/uapi
cflags-y += -Icore/include/linux_headers/arch_header/uapi
cflags-y += -Icore/include/linux_headers/arch_header/generated/uapi
cflags-y += -Icore/include/linux_headers/linux_include

GDEV_SRC_DIRS += lib/user/usched

srcs-y += lib/user/gdev/gdev_lib.c

srcs-y += lib/user/nouveau/nouveau_gdev.c lib/user/nouveau/libnouveau.c lib/user/nouveau/libnouveau_ib.c

srcs-y += common/gdev_api.c common/gdev_device.c common/gdev_sched.c
srcs-y += common/gdev_nvidia.c common/gdev_nvidia_fifo.c common/gdev_nvidia_compute.c
srcs-y += common/gdev_nvidia_mem.c common/gdev_nvidia_shm.c common/gdev_nvidia_nvc0.c common/gdev_nvidia_nve4.c

cflags-y += -DGDEV_SCHED_DISABLED