#include "relu.hpp"
#include "relu_kernel.h"

ReLuLayer::ReLuLayer(int inputOutput) {
    this->input = this->output = inputOutput;
    this->weights = NULL;
    this->bias = NULL;
    this->deltaWeights = NULL;
    this->deltaBias = NULL;

    // Prepare output for forward and backprop
    this->outputForward = NULL;
    this->outputBackward = NULL;
}

Tensor2D* ReLuLayer::forward(Tensor2D* data) {
    this->inputData = data;

    if (!this->outputForward) {
        this->outputForward = new Tensor2D(data->getSize(X), data->getSize(Y));
    }

    dim3 threadsPerBlock(Configuration::reLuBlockSize, Configuration::reLuBlockSize);
    dim3 numBlocks((data->getSize(X) + threadsPerBlock.x)/threadsPerBlock.x,
                   (data->getSize(Y) + threadsPerBlock.y)/threadsPerBlock.y);
        kReLu_kernel(threadsPerBlock, numBlocks,
        data->getDeviceData(), data->getSize(X), data->getSize(Y),
        this->outputForward->getDeviceData()
    );
    return this->outputForward;
}
 
Tensor2D* ReLuLayer::backward(Tensor2D* gradients) {
    dim3 threadsPerBlock(Configuration::reLuBlockSize, Configuration::reLuBlockSize);
    dim3 numBlocks((gradients->getSize(X) + threadsPerBlock.x)/threadsPerBlock.x,
                   (gradients->getSize(Y) + threadsPerBlock.y)/threadsPerBlock.y);
        kReLu_kernel(numBlocks, threadsPerBlock,
        gradients->getDeviceData(), gradients->getSize(X), gradients->getSize(Y),
        gradients->getDeviceData()
    );
    return new Tensor2D(gradients->getSize(X), gradients->getSize(Y), gradients->getDeviceData());
}
