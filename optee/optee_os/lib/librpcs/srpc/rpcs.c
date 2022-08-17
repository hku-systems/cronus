
#include <tee_internal_api.h>
#include <stdio.h>

#include "rpc/crypto.h"
#include "rpc/rpc.h"

extern void *mmap(void *start, size_t len, int prot, int flags, int fd, long off);

static int aes_server_is_init = 0;

int buffer_size_in_bytes;

static char* rbuffer;
static long execution_cnt = 0;
int rpc_entry(char* dummy_buf, int size) {
    buffer_size_in_bytes = 16 * 1024 * 1024;
    rbuffer = (char*) mmap(0, buffer_size_in_bytes, 0x1 | 0x2, 0x1, -1, 0);
    if (!rbuffer) {
        fprintf(stderr, "create buffer failed\n");
        return 1;
    }
    fprintf(stderr, "init handler succeed\n");
    rpc_header_t *header = (rpc_header_t*)rbuffer;
    while (header->is_running != IS_RUNNING_STOP || !execution_cnt) {
        while (header->status == STATUS_START) {
            header->status = STATUS_EXECUTE;
            rpc_dispatch(rbuffer + sizeof(rpc_header_t));
            header->status = STATUS_FIN;
            execution_cnt += 1;
        }
    }
    return 0;
}

// register a handler and buffer for rpc
// int rpc_register(void* buf, rpc_handler handler);

// run the rpc loop
// int rpc_run();