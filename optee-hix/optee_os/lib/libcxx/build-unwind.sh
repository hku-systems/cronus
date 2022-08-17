#!/bin/bash

CUR=$(pwd)

mkdir -p build-unwind
cd build-unwind

CC=$CUR/../../../toolchains/aarch64/bin/aarch64-linux-gnu-gcc CXX=$CUR/../../../toolchains/aarch64/bin/aarch64-linux-gnu-g++ cmake -DLIBUNWIND_ENABLE_SHARED=OFF ../libunwind/
make
