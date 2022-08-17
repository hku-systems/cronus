#include "crossentropy.hpp"
#include "crossentropy_kernel.h"

CrossEntropyLoss::CrossEntropyLoss() {}

float CrossEntropyLoss::getLoss(Tensor2D* networkOutput, Tensor2D* labels) {
    float error = 0.0;
    float* dError;
    cudaMalloc((void**)&dError, sizeof(float));
    cudaMemcpy(dError, &error, sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = Configuration::crossEntropyGetMetricBlockSize;
    dim3 numBlocks((networkOutput->getSize(Y) + threadsPerBlock.x)/threadsPerBlock.x);
        kSoftMaxCrossEntropyLoss_kernel(numBlocks, threadsPerBlock, networkOutput->getDeviceData(), networkOutput->getSize(X), networkOutput->getSize(Y), labels->getDeviceData(), dError);
    cudaMemcpy(&error, dError, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dError);
    return error / networkOutput->getSize(Y);
}

float CrossEntropyLoss::getAccuracy(Tensor2D* networkOutput, Tensor2D* labels) {
    float accuracy = 0.0;
    float* dAccuracy;
    cudaMalloc((void**)&dAccuracy, sizeof(float));
    cudaMemcpy(dAccuracy, &accuracy, sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = Configuration::crossEntropyGetMetricBlockSize;
    dim3 numBlocks((networkOutput->getSize(Y) + threadsPerBlock.x)/threadsPerBlock.x);
        kSoftMaxCrossEntropyAccuracy_kernel(numBlocks, threadsPerBlock, networkOutput->getDeviceData(), networkOutput->getSize(X), networkOutput->getSize(Y), labels->getDeviceData(), dAccuracy);
    cudaMemcpy(&accuracy, dAccuracy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dAccuracy);
    return 100.0 * accuracy / networkOutput->getSize(Y);
}

Tensor2D* CrossEntropyLoss::calculate(Tensor2D* networkOutput, Tensor2D* labels, Tensor2D* output) {
    dim3 threadsPerBlock = Configuration::crossEntropyCalculateBlockSize;
    dim3 numBlocks((networkOutput->getSize(Y) + threadsPerBlock.x)/threadsPerBlock.x);
        kSoftMaxCrossEntropy_kernel(numBlocks, threadsPerBlock, networkOutput->getDeviceData(), networkOutput->getSize(X), networkOutput->getSize(Y), labels->getDeviceData(), output->getDeviceData());
    return output;
}
