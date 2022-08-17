
#pragma once

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int buffer_size_in_bytes;

typedef struct {
    volatile uint8_t is_running;
    volatile uint8_t status;
    uint32_t size;
} rpc_header_t;

#define IS_RUNNING_UD 0
#define IS_RUNNING_START 1
#define IS_RUNNING_STOP 2

#define STATUS_START    1
#define STATUS_EXECUTE  2
#define STATUS_FIN      3

int rpc_open(void* uuid, int buffer_size_in_mb);
int rpc_ecall(uint32_t idx, void *ecall_buf, int bufsize);
void rpc_close();
char* rpc_buffer();


typedef void (*rpc_handler)(void*);
// register a handler and buffer for rpc
int rpc_register(void*, rpc_handler);

// run the rpc loop
int rpc_run();

int rpc_dispatch(char*);
int rpc_entry(char*, int);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#ifdef RPC_DEBUG_ENABLE
#define RPC_DEBUG(format, ...) \
    fprintf(stderr, "D/RPC:%s:%d " #format "\n", __FUNCTION__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

#define RPC_SERVER_DEBUG(format, ...) \
    fprintf(stderr, "D/RPC Remote:%s:%d " #format "\n", __FUNCTION__, __LINE__ __VA_OPT__(,) __VA_ARGS__)
#else 
    #define RPC_DEBUG(format, ...)
    #define RPC_SERVER_DEBUG(format, ...)
#endif

#define RPC_CLIENT_INIT_RET(func,mb)                \
	TEE_UUID uuid = CUDA_TA_UUID;                   \
	int ret;                                        \
	rpc_open(&uuid, mb);                            \
	ret = func(sizeof(argv) / sizeof(char*), argv); \
	rpc_close();                                    \
    return ret;