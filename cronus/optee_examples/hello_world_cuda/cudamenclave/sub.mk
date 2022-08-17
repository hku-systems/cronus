global-incdirs-y += include

srcs-y += hello_world_ta.c
# srcs-y += hello.cpp
srcs-y += cukernel.cu

global-incdirs-y += include/c++/v1
global-incdirs-y += include/cxxabi
global-incdirs-y += include/muslc
global-incdirs-y += include/muslc-arch

include ../../cudaenclave.mk

# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
