
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

incdirs-y += ../

cppflags-y += -ffunction-sections -fdata-sections
cxxflags-y += -include vta/vtaconfig.h

srcs-y += vta_runtime_u.cpp
# srcs-y += vta_runtime_local.cpp
# srcs-y += vta_runtime_caller.cpp
