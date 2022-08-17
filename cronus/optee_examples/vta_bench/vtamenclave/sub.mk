global-incdirs-y += include

srcs-y += hello_world_ta.c

srcs-y += vta/common/test_lib.cc
srcs-y += vta/metal_test/metal_test.cc

global-incdirs-y += include/c++/v1
global-incdirs-y += include/cxxabi
global-incdirs-y += include/muslc
global-incdirs-y += include/muslc-arch

cxxflags-y += -include vta/vtaconfig.h

include ../../vtaenclave.mk

# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
