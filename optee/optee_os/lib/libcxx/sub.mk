
global-incdirs-y += build/include

$(eval $(call bash -c "mkdir build-libcxx"))

srcs-y += __dso_handle.cpp

lib-ar-y = lib/libcxx/libcxx.mri
# libfiles += lib/libcxx/build/lib/libc++.a
# libfiles += lib/libcxx/build-abi/lib/libc++abi.a
# libfiles += lib/libcxx/build-unwind/lib/libunwind.a

# cflags-y += -llib/libcxx/build/lib/libc++.a -llib/libcxx/build-abi/lib/libc++abi.a