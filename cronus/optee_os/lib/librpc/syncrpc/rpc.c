
#include <tee_internal_api.h>
#include <stdio.h>

#include "rpc/crypto.h"
#include "rpc/rpc.h"

static char *buffer;
static TEE_TASessionHandle sess;

int buffer_size_in_bytes;

int rpc_open(void* uuid, int buffer_size_in_mb) {
	
	TEE_Result ret = TEE_OpenTASession(uuid, TEE_TIMEOUT_INFINITE, 0, NULL, &sess, NULL);

    buffer_size_in_bytes = buffer_size_in_mb * 1024 * 1024;

	if (ret != TEE_SUCCESS) {
		fprintf(stderr, "open session failed with %lx\n", ret);
		return 1;
	}

    buffer = malloc(buffer_size_in_bytes);

    if (!buffer) {
        fprintf(stderr, "out of memory in rpc\n");
        return 1;
    }

    if (init_crypto(NULL)) {
        fprintf(stderr, "init crypto fail");
        return 1;
    }

    fprintf(stderr, "open session succeed\n");

    return 0;
}

int rpc_ecall(uint32_t idx, void *ecall_buf, int bufsize) {
	uint32_t paramTypes = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT, TEE_PARAM_TYPE_NONE,
					TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
    TEE_Param params[4];
    int mask = 0xf;
    mask = ~mask;
    int total_size = (bufsize + sizeof(uint32_t));
    int ret;

    uint32_t *idx_ptr = (uint32_t*)buffer;
    *idx_ptr = idx;

    void* encrypted_buffer = malloc(total_size);

    if (!encrypted_buffer) {
        fprintf(stderr, "cannot allocate encrypted buffer %d\n", total_size);
        return 1;
    }

    if (encrypt_aes_cfb_128((unsigned char*)buffer, (unsigned char*)encrypted_buffer, total_size)) {
        fprintf(stderr, "error in encryption\n");
        return 1;
    }

    params[0].memref.buffer = encrypted_buffer;
    params[0].memref.size = total_size;

	ret = TEE_InvokeTACommand(sess, TEE_TIMEOUT_INFINITE, 0, paramTypes, params, NULL);

    if (decrypt_aes_cfb_128((unsigned char*)encrypted_buffer, (unsigned char*)buffer, total_size)) {
        fprintf(stderr, "error in decryption\n");
        return 1;
    }

    free(encrypted_buffer);

    return ret;
}

void rpc_close() {
    TEE_CloseTASession(sess);
    deinit_crypto();
}

char* rpc_buffer() {
    return buffer + sizeof(uint32_t);
}

int rpc_handle(void* buffer) {

}
