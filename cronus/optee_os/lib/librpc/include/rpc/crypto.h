
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int init_crypto(char* inputkey);
int encrypt_aes_cfb_128(unsigned char* plaintext, unsigned char* ciphertext, int inputsize);
int decrypt_aes_cfb_128(unsigned char* ciphertext, unsigned char* plaintext, int inputsize);
void deinit_crypto();

#ifdef __cplusplus
}
#endif /* __cplusplus */