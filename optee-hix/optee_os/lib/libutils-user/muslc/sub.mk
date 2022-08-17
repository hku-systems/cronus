
#
# Makefile for musl (requires GNU make)
#
# This is how simple every makefile should be...
# No, I take that back - actually most should be less than half this size.
#
# Use config.mak to override any of the following variables.
# Do not make changes here.
#

srcdir = ./lib/libutils-user/muslc

SAVED_ARCH := $(ARCH)

ARCH := arm
ifeq ($(ta-target),ta_arm64)
ARCH := aarch64
endif

MALLOC_DIR = mallocbget
SRC_DIRS = $(addprefix $(srcdir)/,src/* $(COMPAT_SRC_DIRS))
BASE_GLOBS = $(addsuffix /*.c,$(SRC_DIRS))
ARCH_GLOBS = $(addsuffix /$(ARCH)/*.[csS],$(SRC_DIRS))
BASE_SRCS = $(sort $(wildcard $(BASE_GLOBS)))
ARCH_SRCS = $(sort $(wildcard $(ARCH_GLOBS)))
BASE_OBJS = $(patsubst $(srcdir)/%,%.o,$(basename $(BASE_SRCS)))
ARCH_OBJS = $(patsubst $(srcdir)/%,%.o,$(basename $(ARCH_SRCS)))
REPLACED_OBJS = $(sort $(subst /$(ARCH)/,/,$(ARCH_OBJS)))
ALL_OBJS = $(addprefix obj/, $(filter-out $(REPLACED_OBJS), $(sort $(BASE_OBJS) $(ARCH_OBJS))))

LIBC_OBJS = $(filter obj/src/%,$(ALL_OBJS)) $(filter obj/compat/%,$(ALL_OBJS))
CRT_OBJS = $(filter obj/crt/%,$(ALL_OBJS))

AOBJS = $(LIBC_OBJS) # for libc.a
LOBJS = $(LIBC_OBJS:.o=.lo)
GENH = obj/include/bits/alltypes.h obj/include/bits/syscall.h
GENH_INT = obj/src/internal/version.h
IMPH = $(addprefix $(srcdir)/, src/internal/stdio_impl.h src/internal/pthread_impl.h src/internal/locale_impl.h src/internal/libc.h)

# they cannot be removed using cflags-remove-y, so we are using cflags-y instead
# of incdirs-y to force to use the muslc headers
cflags-remove-y += -Ilib/libutils/isoc/include 
cflags-remove-y += -Ilib/libutils/ext/include 
cflags-remove-y += -Ilib/libmbedtls/include
cflags-remove-y += -Ilib/libmbedtls/mbedtls/include 
cflags-remove-y += -Ilib/libutee/include 
cflags-remove-y += -Ilib/libdl/include

cflags-y += -std=c99 -nostdinc -ffreestanding -fexcess-precision=standard \
			-frounding-math -Wa,--noexecstack -D_XOPEN_SOURCE=700
cflags-y += -Os -pipe -fomit-frame-pointer -fno-unwind-tables \
			-fno-asynchronous-unwind-tables -ffunction-sections \
			-fdata-sections -Wno-pointer-to-int-cast \
			-Werror=implicit-function-declaration -Werror=implicit-int \
			-Werror=pointer-arith -Werror=int-conversion \
			-Werror=discarded-qualifiers \
			-Werror=discarded-array-qualifiers -Waddress -Warray-bounds \
			-Wchar-subscripts -Wduplicate-decl-specifier -Winit-self \
			-Wreturn-type -Wsequence-point -Wstrict-aliasing -Wunused-function \
			-Wunused-label -Wunused-variable  -fPIC -fno-stack-protector -DCRT

cflags-y += -I$(srcdir)/arch/$(ARCH)/
cflags-y += -I$(srcdir)/arch/generic/
# cflags-y += -I$(srcdir)/obj/src/internal
cflags-y += -I$(srcdir)/src/include
cflags-y += -I$(srcdir)/src/internal
# cflags-y += -I$(srcdir)/obj/include
cflags-y += -I$(srcdir)/include

srcs-muslc-y := $(BASE_SRCS:$(srcdir)/%=%)
srcs-musls-y := $(ARCH_SRCS:$(srcdir)/%=%)
srcs-y += $(srcs-muslc-y) $(srcs-musls-y)
srcs-y += src/malloc/$(MALLOC_DIR)/bget_malloc.c

srcs-y += ../trace.c

cflags-remove-bget_malloc.c-y += -Wold-style-definition -Wredundant-decls
cflags-bget_malloc.c-y += -Wno-sign-compare -Wno-cast-align

ARCH := $(SAVED_ARCH)