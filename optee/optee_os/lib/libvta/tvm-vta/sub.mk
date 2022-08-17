
# file(GLOB FSIM_RUNTIME_SRCS ${VTA_HW_PATH}/src/*.cc)
# file(GLOB FSIM_RUNTIME_SRCS vta/runtime/*.cc)
# list(APPEND FSIM_RUNTIME_SRCS ${VTA_HW_PATH}/src/qemu/sim_driver.cc)
# # list(APPEND FSIM_RUNTIME_SRCS ${VTA_HW_PATH}/src/sim/sim_tlpp.cc)
# # list(APPEND FSIM_RUNTIME_SRCS ${VTA_HW_PATH}/src/vmem/virtual_memory.cc)
# # Target lib: vta_fsim
# add_library(vta_fsim SHARED ${FSIM_RUNTIME_SRCS})
# target_include_directories(vta_fsim SYSTEM PUBLIC ${VTA_HW_PATH}/include)

# usermode driver for vta

srcs-y += src/qemu/sim_driver.cc

cxxflags-y += -include vta/vtaconfig.h