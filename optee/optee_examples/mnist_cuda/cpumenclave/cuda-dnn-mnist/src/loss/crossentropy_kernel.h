
#pragma once

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "cuda_runtime.h"

void kSoftMaxCrossEntropy_kernel(dim3 d1, dim3 d2, float *output, int oX, int oY, float* labels, float* y);
void kSoftMaxCrossEntropyLoss_kernel(dim3 d1, dim3 d2, float *output, int oX, int oY, float* labels, float* error);
void kSoftMaxCrossEntropyAccuracy_kernel(dim3 d1, dim3 d2, float *output, int oX, int oY, float* labels, float* accuracy);

#ifdef __cplusplus
}
#endif /* __cplusplus */