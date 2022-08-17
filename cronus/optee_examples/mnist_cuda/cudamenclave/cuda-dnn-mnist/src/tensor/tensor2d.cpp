#include "tensor2d.hpp"
#include "tensor.h"

Tensor2D::Tensor2D(int sizeX, int sizeY) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    if (this->sizeX && this->sizeY) {
        cudaMalloc((void **)&(this->devData), this->sizeX*this->sizeY*sizeof(float));
    } else {
        this->devData = NULL;
    }
}

Tensor2D::Tensor2D(int sizeX, int sizeY, float** hostData) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    if (this->sizeX && this->sizeY) {
        cudaMalloc((void **)&(this->devData), this->sizeX*this->sizeY*sizeof(float));
        cudaMemcpy(this->devData, *hostData, this->sizeX*this->sizeY*sizeof(float), cudaMemcpyHostToDevice);
    } else {
        this->devData = NULL;
    }
}

Tensor2D::Tensor2D(int sizeX, int sizeY, float* devData) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    this->devData = devData;
}

Tensor2D::~Tensor2D() {
    cudaFree(this->devData);
}

int Tensor2D::getSize(Tensor2DAxis axis) {
    if (axis == X) {
        return this->sizeX;
    } else if (axis == Y) {
        return this->sizeY;
    }
    return -1;
}

float* Tensor2D::getDeviceData() {
    return this->devData;
}

float** Tensor2D::fetchDataFromDevice() {
    float** hostData = new float*[this->sizeY];
    *hostData = new float[this->sizeY * this->sizeX];
    for (int i = 1; i < this->sizeY; i++) hostData[i] = hostData[i-1] + this->sizeX;
    cudaMemcpy(*hostData, this->devData, this->sizeX*this->sizeY*sizeof(float), cudaMemcpyDeviceToHost);
    return hostData;
}

void Tensor2D::add(Tensor1D* tensor) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->getSize()) {
        printf("ERROR! Cannot add vector with size %d to matrix %dx%d.\n",
               tensor->getSize(), this->sizeX, this->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(Configuration::tensor2DAddBlockSize, Configuration::tensor2DAddBlockSize);
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
        kAdd1D_kernel(numBlocks, threadsPerBlock, this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}

void Tensor2D::add(Tensor2D* tensor) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->getSize(X) || this->sizeY != tensor->getSize(Y)) {
        printf("ERROR! Cannot add matrix with size %dx%d to matrix %dx%d.\n",
               tensor->getSize(X), tensor->getSize(Y), this->sizeX, this->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(Configuration::tensor2DAddBlockSize, Configuration::tensor2DAddBlockSize);
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
        kAdd2D_kernel(numBlocks, threadsPerBlock, this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}

void Tensor2D::subtract(Tensor2D* tensor) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->getSize(X) || this->sizeY != tensor->getSize(Y)) {
        printf("ERROR! Cannot subtract matrix with size %dx%d to matrix %dx%d.\n",
               tensor->getSize(X), tensor->getSize(Y), this->sizeX, this->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(Configuration::tensor2DSubtractBlockSize, Configuration::tensor2DSubtractBlockSize);
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
        kSubtract2D_kernel(numBlocks, threadsPerBlock, this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}

void Tensor2D::scale(float factor) {
    dim3 threadsPerBlock(Configuration::tensor2DScaleBlockSize, Configuration::tensor2DScaleBlockSize);
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
        kScale2D_kernel(numBlocks, threadsPerBlock, this->getDeviceData(), factor, this->sizeX, this->sizeY);
}

Tensor2D* Tensor2D::multiply(Tensor2D* tensor, Tensor2D* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->getSize(Y)) {
        printf("ERROR! Cannot multiply matrices with shape %dx%d and %dx%d.\n",
               this->sizeX, this->sizeY, tensor->getSize(X), tensor->getSize(Y));
        exit(1);
    }

    // In case of using shared memory, we've got to use dynamic amount of blocks
    if (Configuration::tensor2DMultiplySharedMemory == 1) {
        // Prepare configuration for CUDA kernel
        dim3 threadsPerBlock(Configuration::tensor2DMultiplyBlockSize, Configuration::tensor2DMultiplyBlockSize);
        dim3 numBlocks((tensor->getSize(X) + threadsPerBlock.x)/threadsPerBlock.x,
                       (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
        int sharedMemorySize = 2 * threadsPerBlock.y * threadsPerBlock.x * sizeof(float);

        // Defer calculations on GPU
        kMultiplyWithSharedMemory_kernel(numBlocks, threadsPerBlock, sharedMemorySize, 
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    } else {
        // Prepare configuration for CUDA kernel
        int threadsX = Configuration::tensor2DMultiplyBlockSize;
        int threadsY = Configuration::tensor2DMultiplyBlockSize;
        int blocksX = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (tensor->getSize(X) + threadsX) / threadsX
                       : Configuration::tensor2DMultiplyBlockNumber;
        int blocksY = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (this->sizeY + threadsY) / threadsY
                       : Configuration::tensor2DMultiplyBlockNumber;
        int fieldsPerBlockX = std::max(1, (tensor->getSize(Y) + blocksX) / blocksX);
        int fieldsPerThreadX = std::max(1, (fieldsPerBlockX + threadsX) / threadsX);
        int fieldsPerBlockY = std::max(1, (this->getSize(Y) + blocksY) / blocksY);
        int fieldsPerThreadY = std::max(1, (fieldsPerBlockY + threadsY) / threadsY);
        dim3 threadsPerBlock(threadsX, threadsY);
        dim3 numBlocks(blocksX, blocksY);

        // Defer calculations on GPU
                kMultiply_kernel(numBlocks, threadsPerBlock, 
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    }
    return output;
}

Tensor2D* Tensor2D::multiplyByTransposition(Tensor2D* tensor, Tensor2D* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->getSize(X)) {
        printf("ERROR! Cannot multiply matrix with shape %dx%d by transposition of matrix %dx%d.\n",
               this->sizeX, this->sizeY, tensor->getSize(X), tensor->getSize(Y));
        exit(1);
    }

    // In case of using shared memory, we've got to use dynamic amount of blocks
    if (Configuration::tensor2DMultiplySharedMemory == 1) {
        // Prepare configuration for CUDA kernel
        dim3 threadsPerBlock(Configuration::tensor2DMultiplyBlockSize, Configuration::tensor2DMultiplyBlockSize);
        dim3 numBlocks((tensor->getSize(Y) + threadsPerBlock.x)/threadsPerBlock.x,
                       (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
        int sharedMemorySize = 2 * threadsPerBlock.y * threadsPerBlock.x * sizeof(float);

        // Defer calculations on GPU
        kMultiplyByTranspositionWithSharedMemory_kernel(numBlocks, threadsPerBlock, sharedMemorySize, 
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    } else {
        // Prepare configuration for CUDA kernel
        int threadsX = Configuration::tensor2DMultiplyBlockSize;
        int threadsY = Configuration::tensor2DMultiplyBlockSize;
        int blocksX = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (tensor->getSize(Y) + threadsX) / threadsX
                       : Configuration::tensor2DMultiplyBlockNumber;
        int blocksY = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (this->sizeY + threadsY) / threadsY
                       : Configuration::tensor2DMultiplyBlockNumber;
        int fieldsPerBlockX = std::max(1, (tensor->getSize(Y) + blocksX) / blocksX);
        int fieldsPerThreadX = std::max(1, (fieldsPerBlockX + threadsX) / threadsX);
        int fieldsPerBlockY = std::max(1, (this->getSize(Y) + blocksY) / blocksY);
        int fieldsPerThreadY = std::max(1, (fieldsPerBlockY + threadsY) / threadsY);
        dim3 threadsPerBlock(threadsX, threadsY);
        dim3 numBlocks(blocksX, blocksY);

        // Defer calculations on GPU
                kMultiplyByTransposition_kernel(numBlocks, threadsPerBlock, 
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    }
    return output;
}

Tensor2D* Tensor2D::transposeAndMultiply(Tensor2D* tensor, Tensor2D* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeY != tensor->getSize(Y)) {
        printf("ERROR! Cannot multiply transposition of matrix with shape %dx%d by matrix %dx%d.\n",
               this->sizeX, this->sizeY, tensor->getSize(X), tensor->getSize(Y));
        exit(1);
    }

    // In case of using shared memory, we've got to use dynamic amount of blocks
    if (Configuration::tensor2DMultiplySharedMemory == 1) {
        // Prepare configuration for CUDA kernel
        dim3 threadsPerBlock(Configuration::tensor2DMultiplyBlockSize, Configuration::tensor2DMultiplyBlockSize);
        dim3 numBlocks((tensor->getSize(X) + threadsPerBlock.x)/threadsPerBlock.x,
                       (this->sizeX + threadsPerBlock.y)/threadsPerBlock.y);
        int sharedMemorySize = 2 * threadsPerBlock.y * threadsPerBlock.x * sizeof(float);

        // Defer calculations on GPU
        kTransposeAndMultiplyWithSharedMemory_kernel(numBlocks, threadsPerBlock, sharedMemorySize, 
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    } else {
        // Prepare configuration for CUDA kernel
        int threadsX = Configuration::tensor2DMultiplyBlockSize;
        int threadsY = Configuration::tensor2DMultiplyBlockSize;
        int blocksX = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (tensor->getSize(X) + threadsX) / threadsX
                       : Configuration::tensor2DMultiplyBlockNumber;
        int blocksY = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (this->getSize(X) + threadsY) / threadsY
                       : Configuration::tensor2DMultiplyBlockNumber;
        int fieldsPerBlockX = std::max(1, (tensor->getSize(X) + blocksX) / blocksX);
        int fieldsPerThreadX = std::max(1, (fieldsPerBlockX + threadsX) / threadsX);
        int fieldsPerBlockY = std::max(1, (this->getSize(X) + blocksY) / blocksY);
        int fieldsPerThreadY = std::max(1, (fieldsPerBlockY + threadsY) / threadsY);
        dim3 threadsPerBlock(threadsX, threadsY);
        dim3 numBlocks(blocksX, blocksY);

        // Defer calculations on GPU
                kTransposeAndMultiply_kernel(numBlocks, threadsPerBlock, 
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    }
    return output;
}

Tensor1D* Tensor2D::meanX(Tensor1D* output) {
    int threadsPerBlock = Configuration::tensor2DMeanBlockSize;
    int numBlocks = (this->sizeX + threadsPerBlock)/threadsPerBlock;
        kMeanX_kernel(numBlocks, threadsPerBlock, this->getDeviceData(), this->sizeX, this->sizeY, output->getDeviceData());
    return output;
}

void Tensor2D::debugPrint() {
    float** values = this->fetchDataFromDevice();
    for (int y = 0; y < this->sizeY; y++) {
        for (int x = 0; x < this->sizeX; x++) {
            printf("%8.5f; ", values[y][x]);
        }
        printf("\n");
    }
    delete[] values;
}
