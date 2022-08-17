# Makefile
TARGET	= user_test
CC	= gcc
SM_ARCH ?= sm_75
NVCC	= nvcc -arch $(SM_ARCH) -cubin
LIBS	= -lucuda -lgdev
CFLAGS	= -L /usr/local/gdev/lib64 -I /usr/local/gdev/include

all:
	gcc -o $(TARGET) $(CFLAGS) main.c $(LIBS)

clean:
	rm -f $(TARGET) *.cubin ./*~
