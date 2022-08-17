
incdirs-y += include

subdirs-y += arch/$(ARCH)/

srcs-y += consttime_memcmp.c
# srcs-y += malloc_ext.c
srcs-y += mempool.c
srcs-y += memzero_explicit.c
srcs-y += printk.c
