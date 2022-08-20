cmake_minimum_required(VERSION 3.13)  # CMake version check
project(simple_example)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 17)            # Enable c++14 standard

set (CMAKE_CXX_FLAGS "-O3 -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl -mavx512cd -mavx512ifma -msse2 -mavx -march=native -ffast-math")

include_directories(./)
# Add main.cpp file of project root directory as source file
set(SOURCE_FILES sim_driver.cc sim_tlpp.cc vmem/virtual_memory.cc)

# Add executable target with source files listed in SOURCE_FILES variable
add_library(vta_device SHARED ${SOURCE_FILES})
target_link_libraries(vta_device glog)

add_executable(vta_test test/test_device.c)
target_link_libraries(vta_test dl)