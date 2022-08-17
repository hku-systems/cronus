
srcs-y += driver/context.c
srcs-y += driver/device.c
srcs-y += driver/dummy.c
srcs-y += driver/event.c
srcs-y += driver/execution.c
srcs-y += driver/gdev_cuda.c
srcs-y += driver/init.c
srcs-y += driver/memory.c
srcs-y += driver/module.c
srcs-y += driver/stream.c
srcs-y += driver/version.c

srcs-y += runtime/ocelot/cuda/implementation/cuda_runtime.cpp
srcs-y += runtime/ocelot/cuda/implementation/CudaDriver.cpp
srcs-y += runtime/ocelot/cuda/implementation/CudaDriverFrontend.cpp
srcs-y += runtime/ocelot/cuda/implementation/CudaDriverInterface.cpp
srcs-y += runtime/ocelot/cuda/implementation/CudaRuntime.cpp
srcs-y += runtime/ocelot/cuda/implementation/CudaRuntimeInterface.cpp
srcs-y += runtime/ocelot/cuda/implementation/CudaWorkerThreadFake.cpp
srcs-y += runtime/ocelot/cuda/implementation/FatBinaryContext.cpp

srcs-y += runtime/ocelot/executive/implementation/Device.cpp
srcs-y += runtime/ocelot/executive/implementation/ExecutableKernel.cpp
srcs-y += runtime/ocelot/executive/implementation/FrameInfo.cpp
srcs-y += runtime/ocelot/executive/implementation/NVIDIAExecutableKernel.cpp
srcs-y += runtime/ocelot/executive/implementation/NVIDIAGPUDevice.cpp
srcs-y += runtime/ocelot/executive/implementation/RuntimeException.cpp

srcs-y += runtime/ocelot/ir/implementation/ControlFlowGraph.cpp
srcs-y += runtime/ocelot/ir/implementation/Dim3.cpp
srcs-y += runtime/ocelot/ir/implementation/Global.cpp
srcs-y += runtime/ocelot/ir/implementation/Instruction.cpp
srcs-y += runtime/ocelot/ir/implementation/IRKernel.cpp
srcs-y += runtime/ocelot/ir/implementation/Kernel.cpp
srcs-y += runtime/ocelot/ir/implementation/Local.cpp
srcs-y += runtime/ocelot/ir/implementation/Module.cpp
srcs-y += runtime/ocelot/ir/implementation/Parameter.cpp
srcs-y += runtime/ocelot/ir/implementation/PTXInstruction.cpp
srcs-y += runtime/ocelot/ir/implementation/PTXKernel.cpp
srcs-y += runtime/ocelot/ir/implementation/PTXOperand.cpp
srcs-y += runtime/ocelot/ir/implementation/PTXStatement.cpp
srcs-y += runtime/ocelot/ir/implementation/Texture.cpp
srcs-y += $(OCELOT_IR_FILES)

srcs-y += runtime/hydrazine/implementation/Exception.cpp
srcs-y += runtime/hydrazine/implementation/SystemCompatibility.cpp
srcs-y += runtime/hydrazine/implementation/Timer.cpp
srcs-y += runtime/hydrazine/implementation/LowLevelTimer.cpp
srcs-y += runtime/hydrazine/implementation/Version.cpp
srcs-y += runtime/hydrazine/implementation/debug.cpp
srcs-y += runtime/hydrazine/implementation/string.cpp

srcs-y += runtime/ocelot/transforms/implementation/SharedPtrAttribute.cpp
srcs-y += runtime/ocelot/analysis/implementation/DataflowGraph.cpp

incdirs-y += .
incdirs-y += libucuda
incdirs-y += runtime
incdirs-y += driver
incdirs-y += ../gen
incdirs-y += ../common
incdirs-y += ../util
incdirs-y += ../lib/user
incdirs-y += ../lib/user/gdev
incdirs-y += ../lib/user/nouveau
incdirs-y += ../../../../core/include/linux_headers/linux_include/generated/uapi/

cppflags-y += -ffunction-sections -fdata-sections -DENCLAVE -DENABLE_CUBIN_MODULE -DGDEV_SCHED_DISABLED