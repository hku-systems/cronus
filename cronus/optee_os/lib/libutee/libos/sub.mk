
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

srcs-y += tee_syscall_musl.c
srcs-y += tee_sys_fs.c
srcs-y += tee_id.c
srcs-y += tee_sched.c
srcs-y += tee_mm.c
srcs-y += tee_time.c
srcs-y += libos.c