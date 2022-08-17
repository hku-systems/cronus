
#include "relu_kernel.h"

__global__
void kReLu(float *A, int aX, int aY, float* B) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < aX && y < aY) {
        if (A[y*aX + x] < 0.0) {
            B[y*aX + x] = 0;
        } else {
            B[y*aX + x] = A[y*aX + x];
        }
    }
}

extern "C" void kReLu_kernel(dim3 numBlocks, dim3 threadsPerBlock, float *A, int aX, int aY, float* B) {
        kReLu<<<numBlocks, threadsPerBlock>>>(A, aX, aY, B);
}