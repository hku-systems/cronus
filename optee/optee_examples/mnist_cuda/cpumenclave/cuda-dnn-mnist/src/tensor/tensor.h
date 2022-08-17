
#pragma once

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "cuda_runtime.h"

void kAdd_kernel(dim3 d1, dim3 d2, float *a, float *b, int N);
void kSubtract1D_kernel(dim3 d1, dim3 d2, float *a, float *b, int N);
void kAdd1D_kernel(dim3 d1, dim3 d2, float *a, float *b, int sizeX, int sizeY);
void kScale1D_kernel(dim3 d1, dim3 d2, float *a, float factor, int N);

void kAdd2D_kernel(dim3 d1, dim3 d2, float *a, float *b, int sizeX, int sizeY);
void kSubtract2D_kernel(dim3 d1, dim3 d2, float *a, float *b, int sizeX, int sizeY);
void kScale2D_kernel(dim3 d1, dim3 d2, float *a, float factor, int sizeX, int sizeY);
void kMultiply_kernel(dim3 d1, dim3 d2, int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
               float* A, int aX, int aY,
               float* B, int bX, int bY,
               float* C);
void kMultiplyWithSharedMemory_kernel(dim3 d1, dim3 d2, int d3, float* A, int aX, int aY,
                               float* B, int bX, int bY,
                               float* C);

void kMultiplyByTransposition_kernel(dim3 d1, dim3 d2, int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
                              float* A, int aX, int aY,
                              float* B, int bX, int bY,
                              float* C);

void kMultiplyByTranspositionWithSharedMemory_kernel(dim3 d1, dim3 d2, int d3, float* A, int aX, int aY,
                                              float* B, int bX, int bY,
                                              float* C);

void kTransposeAndMultiply_kernel(dim3 d1, dim3 d2, int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
                           float* A, int aX, int aY,
                           float* B, int bX, int bY,
                           float* C);

void kTransposeAndMultiplyWithSharedMemory_kernel(dim3 d1, dim3 d2, int d3, float* A, int aX, int aY,
                                           float* B, int bX, int bY,
                                           float* C);

void kMeanX_kernel(dim3 d1, dim3 d2, float* a, int aX, int aY, float* b);

#ifdef __cplusplus
}
#endif /* __cplusplus */