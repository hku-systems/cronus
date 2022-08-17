
#include "crossentropy_kernel.h"

#define VERY_SMALL_NUMBER 1e-10

__global__
void kSoftMaxCrossEntropy(float *output, int oX, int oY, float* labels, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < oY) {
        // Calculate sum of exponents for whole column
        float sum = 0.0;
        for (int i = 0; i < oX; i++) {
            sum += exp(output[row*oX + i]);
        }
        if (abs(sum) < VERY_SMALL_NUMBER) {
            sum = VERY_SMALL_NUMBER;
        }

        // Softmax = exp(value) / sum(exp(allValues))
        // Subtract truth (which is one hot)
        for (int i = 0; i < oX; i++) {
            y[row*oX + i] = (exp(output[row*oX + i]) / sum) - labels[row*oX + i];
        }
    }
}

__global__
void kSoftMaxCrossEntropyLoss(float *output, int oX, int oY, float* labels, float* error) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < oY) {
        // Calculate sum of exponents for whole column
        float sum = 0.0;
        for (int i = 0; i < oX; i++) {
            sum += exp(output[row*oX + i]);
        }
        if (abs(sum) < VERY_SMALL_NUMBER) {
            sum = VERY_SMALL_NUMBER;
        }

        // Error = target * log(softmaxOutput) + (1 - target) * log(1 - softmaxOutput)
        float tmpError = 0.0;
        for (int i = 0; i < oX; i++) {
            float softmaxOutput = exp(output[row*oX + i]) / sum;
            tmpError -= labels[row*oX + i] * log(softmaxOutput) + 
                        (1 - labels[row*oX + i]) * log(1 - softmaxOutput);
        }
        atomicAdd(error, tmpError);
    }
}

__global__
void kSoftMaxCrossEntropyAccuracy(float *output, int oX, int oY, float* labels, float* accuracy) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < oY) {
        int maxIdx = 0;
        float maxValue = output[row*oX];
        for (int x = 1; x < oX; x++) {
            if (output[row*oX + x] > maxValue) {
                maxIdx = x;
                maxValue = output[row*oX + x];
            }
        }
        if (output[row*oX + maxIdx] > 1.0 - VERY_SMALL_NUMBER) {
            atomicAdd(accuracy, 1);
        }
    }
}

extern "C" void kSoftMaxCrossEntropy_kernel(dim3 numBlocks, dim3 threadsPerBlock, float *output, int oX, int oY, float* labels, float* y) {
        kSoftMaxCrossEntropy<<<numBlocks, threadsPerBlock>>>(output, oX, oY, labels, y);
}

extern "C" void kSoftMaxCrossEntropyLoss_kernel(dim3 numBlocks, dim3 threadsPerBlock, float *output, int oX, int oY, float* labels, float* error) {
        kSoftMaxCrossEntropyLoss<<<numBlocks, threadsPerBlock>>>(output, oX, oY, labels, error);
}

extern "C" void kSoftMaxCrossEntropyAccuracy_kernel(dim3 numBlocks, dim3 threadsPerBlock, float *output, int oX, int oY, float* labels, float* accuracy) {
        kSoftMaxCrossEntropyAccuracy<<<numBlocks, threadsPerBlock>>>(output, oX, oY, labels, accuracy);
}