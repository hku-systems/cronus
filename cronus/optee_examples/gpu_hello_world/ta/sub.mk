global-incdirs-y += include
srcs-y += hello_world_ta.c
srcs-y += matrixadd.c
srcs-y += main.c

libnames += drm
libnames += gdev

# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
