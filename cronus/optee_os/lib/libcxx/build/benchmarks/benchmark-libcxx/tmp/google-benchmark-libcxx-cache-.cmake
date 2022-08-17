
set(CMAKE_C_COMPILER "/home/jianyu/optee/toolchains/aarch64/bin/aarch64-none-linux-gnu-gcc" CACHE STRING "Initial cache" FORCE)
set(CMAKE_CXX_COMPILER "/home/jianyu/optee/toolchains/aarch64/bin/aarch64-none-linux-gnu-g++" CACHE STRING "Initial cache" FORCE)
set(CMAKE_BUILD_TYPE "RELEASE" CACHE STRING "Initial cache" FORCE)
set(CMAKE_INSTALL_PREFIX "/home/jianyu/optee/optee_os/lib/libcxx/build/benchmarks/benchmark-libcxx" CACHE PATH "Initial cache" FORCE)
set(CMAKE_CXX_FLAGS "-Wno-unused-command-line-argument -nostdinc++ -isystem /home/jianyu/optee/optee_os/lib/libcxx/build/include/c++/v1 -L/home/jianyu/optee/optee_os/lib/libcxx/build/lib -Wl,-rpath,/home/jianyu/optee/optee_os/lib/libcxx/build/lib -L -Wl,-rpath," CACHE STRING "Initial cache" FORCE)
set(BENCHMARK_USE_LIBCXX "ON" CACHE BOOL "Initial cache" FORCE)
set(BENCHMARK_ENABLE_TESTING "OFF" CACHE BOOL "Initial cache" FORCE)