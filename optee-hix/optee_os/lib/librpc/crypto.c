
#include <mbedtls/aes.h>

static unsigned char iv[] = {0xff, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
static unsigned char key[] = {0xff, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0xff, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};

static mbedtls_aes_context aes;

int init_crypto(char* inputkey) {
    mbedtls_aes_init(&aes);
    return mbedtls_aes_setkey_enc(&aes, key, 256);
}

int encrypt_aes_cfb_128(unsigned char* plaintext, unsigned char* ciphertext, int inputsize) {
    size_t iv_offset = 0;
    return mbedtls_aes_crypt_cfb128(&aes, MBEDTLS_AES_ENCRYPT, inputsize, &iv_offset, iv, plaintext, ciphertext);
}

int decrypt_aes_cfb_128(unsigned char* ciphertext, unsigned char* plaintext, int inputsize) {
    size_t iv_offset1 = 0;
    return mbedtls_aes_crypt_cfb128(&aes, MBEDTLS_AES_DECRYPT, inputsize, &iv_offset1, iv, ciphertext, plaintext);
}

void deinit_crypto() {
    mbedtls_aes_free(&aes);
}