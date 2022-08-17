# Makefile
TARGET	= user_test
CC	= gcc
SM_ARCH ?= sm_75
NVCC	= nvcc -arch $(SM_ARCH) -cubin
LIBS	= -lucuda -lgdev
CFLAGS	= -L /usr/local/gdev/lib64 -I /usr/local/gdev/include

all:
	$(NVCC) -o fmmul_gpu.cubin fmmul_gpu.cu
	gcc -o $(TARGET) $(CFLAGS) main.c fmmul.c $(LIBS)

clean:
	rm -f $(TARGET) *.cubin ./*~
