global-incdirs-y += include

srcs-y += hello_world_ta.c

# srcs-y += cuda/backprop/backprop.c
# srcs-y += cuda/backprop/facetrain.c
# srcs-y += cuda/backprop/imagenet.c
# srcs-y += cuda/backprop/backprop_cuda.cu
# srcs-y += cuda/backprop/backprop_cuda_kernel.cu

# srcs-y += cuda/bfs/bfs.cu
# srcs-y += cuda/bfs/kernel.cu
# srcs-y += cuda/bfs/kernel2.cu

# srcs-y += cuda/gaussian/gaussian.cu

# srcs-y += cuda/hotspot/hotspot.cu

# srcs-y += cuda/lud/common/common.c

# srcs-y += cuda/lud/cuda/lud_kernel.cu
# srcs-y += cuda/lud/cuda/lud.cu

# srcs-y += cuda/nw/needle.cu
# srcs-y += cuda/nw/needle_kernel.cu

# srcs-y += cuda/nn/nn_cuda.cu

# srcs-y += cuda/pathfinder/pathfinder.cu

srcs-y += cuda/srad/srad_v2/srad.cu
# srcs-y += cuda/srad/srad_v2/srad_kernel.cu

global-incdirs-y += include/c++/v1
global-incdirs-y += include/cxxabi
global-incdirs-y += include/muslc
global-incdirs-y += include/muslc-arch

include ../../cudaenclave.mk

# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
