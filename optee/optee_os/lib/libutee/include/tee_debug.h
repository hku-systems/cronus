

#ifndef TEE_DEBUG_H
#define TEE_DEBUG_H

#include <sys/time.h>

extern long long elapsed_time_ms;

#define TEE_TIME_START struct timeval tvs, tve; \
    elapsed_time_ms = 0; \
    gettimeofday(&tvs, 0);

#define TEE_TIME_END gettimeofday(&tve, 0); \
    elapsed_time_ms = (tve.tv_sec - tvs.tv_sec) * 1000 + (tve.tv_usec - tvs.tv_usec) / 1000;

#endif