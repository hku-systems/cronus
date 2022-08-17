# From utils/ext
# These files implements the__aeabi functions we need instead of
# relying on libgcc or equivalent as we need implementations suitable
# for bare metal.
srcs-$(CFG_ARM32_$(sm)) += arm32_aeabi_divmod_a32.S
srcs-$(CFG_ARM32_$(sm)) += arm32_aeabi_divmod.c
srcs-$(CFG_ARM32_$(sm)) += arm32_aeabi_ldivmod_a32.S
srcs-$(CFG_ARM32_$(sm)) += arm32_aeabi_ldivmod.c
srcs-$(CFG_ARM32_$(sm)) += arm32_aeabi_shift.c

ifeq ($(CFG_ULIBS_MCOUNT),y)
# We would not like to profile __aeabi functions as these provide
# internal implementations for "/ %" operations. Also, "/ %" operations
# could be used inside profiling code which could create an incorrect
# cyclic behaviour.
cflags-remove-arm32_aeabi_divmod.c-y += -pg
cflags-remove-arm32_aeabi_ldivmod.c-y += -pg
cflags-remove-arm32_aeabi_shift.c-y += -pg
endif

# from utils/isoc
ifeq ($(CFG_UNWIND),y)
srcs-$(CFG_ARM32_$(sm)) += aeabi_unwind.c
endif
srcs-$(CFG_ARM32_$(sm)) += atomic_a32.S
srcs-$(CFG_ARM64_$(sm)) += atomic_a64.S
srcs-y += auxval.c
ifneq ($(sm),ldelf) # TA, core
srcs-$(CFG_ARM32_$(sm)) += mcount_a32.S
srcs-$(CFG_ARM64_$(sm)) += mcount_a64.S
endif
