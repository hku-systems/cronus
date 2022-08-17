
#include <tee_internal_api.h>
#include <stdio.h>

#include "rpc/crypto.h"
#include "rpc/rpc.h"

static char *buffer;
static rpc_header_t *header;
static TEE_TASessionHandle sess;

int buffer_size_in_bytes;

extern void *mmap(void *start, size_t len, int prot, int flags, int fd, long off);

int rpc_open(void* uuid, int buffer_size_in_mb) {
	
    buffer_size_in_bytes = buffer_size_in_mb * 1024 * 1024;

    buffer = mmap(0, buffer_size_in_bytes, 0x1 | 0x2, 0x1, -1, 0);

    if (!buffer) {
        fprintf(stderr, "out of memory in rpc\n");
        return 1;
    }

    header = (rpc_header_t*) buffer;
    header->is_running = IS_RUNNING_START;

    fprintf(stderr, "open session succeed\n");

    return 0;
}

int rpc_ecall(uint32_t idx, void *ecall_buf, int bufsize) {
    int total_size = (bufsize + sizeof(uint32_t));
    int ret;

    uint32_t *idx_ptr = (uint32_t*)(buffer + sizeof(rpc_header_t));
    *idx_ptr = idx;

    header->size = total_size;
    header->status = STATUS_START;
    while (header->status != STATUS_FIN);

    return ret;
}

void rpc_close() {
    // do unmap
    header->is_running = IS_RUNNING_STOP;
}

char* rpc_buffer() {
    return buffer + sizeof(rpc_header_t) + sizeof(uint32_t);
}

int rpc_handle(void* buffer) {

}
