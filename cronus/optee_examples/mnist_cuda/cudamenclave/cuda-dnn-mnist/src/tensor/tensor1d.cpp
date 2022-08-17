#include "tensor1d.hpp"
#include "tensor.h"


Tensor1D::Tensor1D(int size) {
    this->size = size;
    if (size) {
        cudaMalloc((void **)&(this->devData), this->size*sizeof(float));
    } else {
        this->devData = NULL;
    }
}

Tensor1D::Tensor1D(int size, float* hostData) {
    this->size = size;
    if (size) {
        cudaMalloc((void **)&(this->devData), this->size*sizeof(float));
        cudaMemcpy(this->devData, hostData, this->size*sizeof(float), cudaMemcpyHostToDevice);
    } else {
        this->devData = NULL;
    }
}

Tensor1D::~Tensor1D() {
    cudaFree(this->devData);
}

int Tensor1D::getSize() {
    return this->size;
}

float* Tensor1D::getDeviceData() {
    return this->devData;
}

float* Tensor1D::fetchDataFromDevice() {
    float* hostData = new float[this->size];
    cudaDeviceSynchronize();
    cudaMemcpy(hostData, this->devData, this->size*sizeof(float), cudaMemcpyDeviceToHost);
    return hostData;
}

void Tensor1D::add(Tensor1D* tensor) {
        kAdd_kernel(this->size, 1, this->getDeviceData(), tensor->getDeviceData(), this->size);
}

void Tensor1D::subtract(Tensor1D* tensor) {
        kSubtract1D_kernel(this->size, 1, this->getDeviceData(), tensor->getDeviceData(), this->size);
}

void Tensor1D::scale(float factor) {
        kScale1D_kernel(this->size, 1, this->getDeviceData(), factor, this->size);
}
