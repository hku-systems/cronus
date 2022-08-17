
#include <tee_internal_api.h>
#include <stdio.h>

#include "rpc/crypto.h"
#include "rpc/rpc.h"

static int aes_server_is_init = 0;

int rpc_entry(char* encrypted_buffer, int size) {
    if (!aes_server_is_init) {
        aes_server_is_init = 1;
        init_crypto(NULL);
    }

    void* plaintext = malloc(size);

    if (!plaintext) {
        fprintf(stderr, "error in allocating buffer\n");
        return 1;
    }

    if (decrypt_aes_cfb_128((unsigned char*)encrypted_buffer, plaintext , size)) {
        fprintf(stderr, "error in encryption\n");
        return 1;
    }
    if (rpc_dispatch(plaintext)) {
        free(plaintext);
        return 1;
    }
    if (encrypt_aes_cfb_128(plaintext, (unsigned char*)encrypted_buffer, size)) {
        fprintf(stderr, "error in encryption\n");
        return 1;
    }
}

// register a handler and buffer for rpc
// int rpc_register(void* buf, rpc_handler handler);

// run the rpc loop
// int rpc_run();