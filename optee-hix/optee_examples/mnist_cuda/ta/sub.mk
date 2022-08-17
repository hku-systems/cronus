global-incdirs-y += include

srcs-y += hello_world_ta.c
srcs-y += hello.cpp

# cpp files
srcs-y += cuda-dnn-mnist/src/optimizers/sgd.cpp
srcs-y += cuda-dnn-mnist/src/utils.cpp
srcs-y += cuda-dnn-mnist/src/layers/layer.cpp
srcs-y += cuda-dnn-mnist/src/layers/dense.cpp
srcs-y += cuda-dnn-mnist/src/datasets/mnist.cpp
srcs-y += cuda-dnn-mnist/src/loggers/csv_logger.cpp
srcs-y += cuda-dnn-mnist/src/models/sequential.cpp
srcs-y += cuda-dnn-mnist/src/configuration.cpp

# cu files
srcs-y += cuda-dnn-mnist/src/main.cu
srcs-y += cuda-dnn-mnist/src/layers/relu.cu
srcs-y += cuda-dnn-mnist/src/tensor/tensor1d.cu
srcs-y += cuda-dnn-mnist/src/tensor/tensor2d.cu
srcs-y += cuda-dnn-mnist/src/loss/crossentropy.cu


global-incdirs-y += include/c++/v1
global-incdirs-y += include/cxxabi
global-incdirs-y += include/muslc
global-incdirs-y += include/muslc-arch
global-incdirs-y += gdev-cuda/cuda/runtime/ocelot/cuda/interface/
global-incdirs-y += gdev-cuda/cuda/runtime

libnames += drm
libnames += gdev
libnames += cxx

subdirs-y += gdev-cuda/cuda/

# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
