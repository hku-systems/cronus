#!/bin/bash

CUR=$(pwd)

mkdir -p build-abi
cd build-abi

CXX=$CUR/../../../toolchains/aarch64/bin/aarch64-linux-gnu-g++ cmake -DLIBCXXABI_LIBCXX_INCLUDES=../build/include/c++/v1/ -DLIBCXXABI_ENABLE_THREADS=OFF -DLIBCXXABI_ENABLE_SHARED=OFF ../libcxxabi/
make
