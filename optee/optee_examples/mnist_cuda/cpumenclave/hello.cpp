
#include "hello.h"
#include <list>
#include <stdbool.h>
#include <inttypes.h>


extern void (*__init_array_start []) (void) __attribute__((weak));
extern void (*__init_array_end []) (void) __attribute__((weak));

#include <iostream>

void init_cpp_rt() {
	size_t count;
    count = (size_t)(__init_array_end - __init_array_start);
    for (int i = 0; i < count; i++)
        __init_array_start[i]();	
}

extern "C" int compute_out(int a, int b) {
	init_cpp_rt();
	return a + b;
}
