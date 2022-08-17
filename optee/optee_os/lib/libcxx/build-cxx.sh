#!/bin/bash

CUR=$(pwd)

mkdir -p build
cd build

CXX=$CUR/../../../toolchains/aarch64/bin/aarch64-linux-gnu-g++ cmake -DLIBCXX_MUSL_LIBC_DIR=$CUR/../libutils-user/muslc/ -DLIBCXX_ENABLE_SHARED=OFF -DLIBCXX_ENABLE_THREADS=OFF -DLIBCXX_HAS_MUSL_LIBC=ON -DLIBCXX_ENABLE_MONOTONIC_CLOCK=OFF -DHAVE_LIBCXXABI=ON ../libcxx
make
