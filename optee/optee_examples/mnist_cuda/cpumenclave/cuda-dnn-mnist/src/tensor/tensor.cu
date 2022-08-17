
#include "tensor.h"

__global__
void kAdd(float *a, float *b, int N) {
    int i = blockIdx.x;
    if (i < N) {
        a[i] += b[i];
    }
}

__global__
void kSubtract1D(float *a, float *b, int N) {
    int i = blockIdx.x;
    if (i < N) {
        a[i] -= b[i];
    }
}

__global__
void kAdd1D(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] += b[x];
    }
}

__global__
void kScale1D(float *a, float factor, int N) {
    int i = blockIdx.x;
    if (i < N) {
        a[i] *= factor;
    }
}

__global__
void kAdd2D(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] += b[y*sizeX + x];
    }
}

__global__
void kSubtract2D(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] -= b[y*sizeX + x];
    }
}

__global__
void kScale2D(float *a, float factor, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] *= factor;
    }
}

__global__
void kMultiply(int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
               float* A, int aX, int aY,
               float* B, int bX, int bY,
               float* C)
{
    int outputSizeX = bX;
    int outputSizeY = aY;
    int blockStartX = blockIdx.x * fieldsPerBlockX;
    int blockStartY = blockIdx.y * fieldsPerBlockY;
    int blockEndX = min(outputSizeX, blockStartX + fieldsPerBlockX);
    int blockEndY = min(outputSizeY, blockStartY + fieldsPerBlockY);
    int threadStartX = threadIdx.x * fieldsPerThreadX;
    int threadStartY = threadIdx.y * fieldsPerThreadY;
    int threadEndX = threadStartX + fieldsPerThreadX;
    int threadEndY = threadStartY + fieldsPerThreadY;

    int startX = blockStartX + threadStartX;
    int endX = min(blockEndX, blockStartX + threadEndX);
    int startY = blockStartY + threadStartY;
    int endY = min(blockEndY, blockStartY + threadEndY);

    for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
            float sum = 0.0f;
            for (int i = 0; i < aX; i++) {
                sum += A[y*aX + i] * B[i*bX + x];
            }
            C[y*bX + x] = sum;
        }
    }
}

__global__
void kMultiplyWithSharedMemory(float* A, int aX, int aY,
                               float* B, int bX, int bY,
                               float* C)
{
    int outputSizeX = bX;
    int outputSizeY = aY;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int chunks = (aX + blockDim.x) / blockDim.x;

    if (x >= outputSizeX || y >= outputSizeY) return;

    extern __shared__ float sub[];
    float* As = sub;
    float* Bs = sub + blockDim.x * blockDim.y;

    float sum = 0.0f;
    for (int chunk = 0; chunk < chunks; chunk++) {
        // Safely copy data from matrix A
        if (chunk * blockDim.x + threadIdx.x < aX && blockIdx.y * blockDim.y + threadIdx.y < aY) {
            As[threadIdx.y * blockDim.x + threadIdx.x] = A[(blockIdx.y * blockDim.y + threadIdx.y) * aX + chunk * blockDim.x + threadIdx.x];
        } else {
            As[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
        }

        // Safely copy data from matrix B
        if (blockIdx.x * blockDim.x + threadIdx.x < bX && chunk * blockDim.y + threadIdx.y < bY) {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(chunk * blockDim.y + threadIdx.y) * bX + blockIdx.x * blockDim.x + threadIdx.x];
        } else {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
        }

        // Run calculations on shared memory matrix
        __syncthreads();
        for (int i = 0; i < blockDim.x; i++) {
            sum += As[threadIdx.y * blockDim.x + i] * Bs[i * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }
    C[y*outputSizeX + x] = sum;
}

__global__
void kMultiplyByTransposition(int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
                              float* A, int aX, int aY,
                              float* B, int bX, int bY,
                              float* C)
{
    int blockStartX = blockIdx.x * fieldsPerBlockX;
    int blockStartY = blockIdx.y * fieldsPerBlockY;
    int blockEndX = min(bY, blockStartX + fieldsPerBlockX);
    int blockEndY = min(aY, blockStartY + fieldsPerBlockY);
    int threadStartX = threadIdx.x * fieldsPerThreadX;
    int threadStartY = threadIdx.y * fieldsPerThreadY;
    int threadEndX = threadStartX + fieldsPerThreadX;
    int threadEndY = threadStartY + fieldsPerThreadY;

    int startX = blockStartX + threadStartX;
    int endX = min(blockEndX, blockStartX + threadEndX);
    int startY = blockStartY + threadStartY;
    int endY = min(blockEndY, blockStartY + threadEndY);

    for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
            float sum = 0.0f;
            for (int i = 0; i < aX; i++) {
                sum += A[y*aX + i] * B[x*bX + i];
            }
            C[y*bY + x] = sum;
        }
    }
}

__global__
void kMultiplyByTranspositionWithSharedMemory(float* A, int aX, int aY,
                                              float* B, int bX, int bY,
                                              float* C)
{
    int outputSizeX = bY;
    int outputSizeY = aY;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int chunks = (aX + blockDim.x) / blockDim.x;

    if (x >= outputSizeX || y >= outputSizeY) return;

    extern __shared__ float sub[];
    float* As = sub;
    float* Bs = sub + blockDim.x * blockDim.y;

    float sum = 0.0f;
    for (int chunk = 0; chunk < chunks; chunk++) {
        // Safely copy data from matrix A
        if (chunk * blockDim.x + threadIdx.x < aX && blockIdx.y * blockDim.y + threadIdx.y < aY) {
            As[threadIdx.y * blockDim.x + threadIdx.x] = A[(blockIdx.y * blockDim.y + threadIdx.y) * aX + chunk * blockDim.x + threadIdx.x];
        } else {
            As[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
        }

        // Safely copy data from matrix B
        if (chunk * blockDim.x + threadIdx.x < bX && blockIdx.x * blockDim.y + threadIdx.y < bY) {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(blockIdx.x * blockDim.y + threadIdx.y) * bX + chunk * blockDim.x + threadIdx.x];
        } else {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
        }

        // Run calculations on shared memory matrix
        __syncthreads();
        for (int i = 0; i < blockDim.x; i++) {
            sum += As[threadIdx.y * blockDim.x + i] * Bs[threadIdx.x * blockDim.x + i];
        }
        __syncthreads();
    }
    C[y*outputSizeX + x] = sum;
}

__global__
void kTransposeAndMultiply(int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
                           float* A, int aX, int aY,
                           float* B, int bX, int bY,
                           float* C)
{
    int blockStartX = blockIdx.x * fieldsPerBlockX;
    int blockStartY = blockIdx.y * fieldsPerBlockY;
    int blockEndX = min(bX, blockStartX + fieldsPerBlockX);
    int blockEndY = min(aX, blockStartY + fieldsPerBlockY);
    int threadStartX = threadIdx.x * fieldsPerThreadX;
    int threadStartY = threadIdx.y * fieldsPerThreadY;
    int threadEndX = threadStartX + fieldsPerThreadX;
    int threadEndY = threadStartY + fieldsPerThreadY;

    int startX = blockStartX + threadStartX;
    int endX = min(blockEndX, blockStartX + threadEndX);
    int startY = blockStartY + threadStartY;
    int endY = min(blockEndY, blockStartY + threadEndY);

    for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
            float sum = 0.0f;
            for (int i = 0; i < bY; i++) {
                sum += A[i*aX + y] * B[i*bX + x];
            }
            C[y*bX + x] = sum;
        }
    }
}

__global__
void kTransposeAndMultiplyWithSharedMemory(float* A, int aX, int aY,
                                           float* B, int bX, int bY,
                                           float* C)
{
    int outputSizeX = bX;
    int outputSizeY = aX;
    int elementsInChunk = blockDim.x;  // X & Y should be equal!
    int x = blockIdx.x * elementsInChunk + threadIdx.x;
    int y = blockIdx.y * elementsInChunk + threadIdx.y;
    int chunks = (aY + elementsInChunk) / elementsInChunk;

    if (x >= outputSizeX || y >= outputSizeY) return;

    extern __shared__ float sub[];
    float* As = sub;
    float* Bs = sub + elementsInChunk * elementsInChunk;

    float sum = 0.0f;
    for (int chunk = 0; chunk < chunks; chunk++) {
        // Safely copy data from matrix A
        if (blockIdx.y * elementsInChunk + threadIdx.x < aX && chunk * elementsInChunk + threadIdx.y < aY) {
            As[threadIdx.y * elementsInChunk + threadIdx.x] = 
                A[(chunk * elementsInChunk + threadIdx.y) * aX + blockIdx.y * elementsInChunk + threadIdx.x];
        } else {
            As[threadIdx.y * elementsInChunk + threadIdx.x] = 0.0;
        }

        // Safely copy data from matrix B
        if (blockIdx.x * elementsInChunk + threadIdx.x < bX && chunk * elementsInChunk + threadIdx.y < bY) {
            Bs[threadIdx.y * elementsInChunk + threadIdx.x] =
                B[(chunk * elementsInChunk + threadIdx.y) * bX + blockIdx.x * elementsInChunk + threadIdx.x];
        } else {
            Bs[threadIdx.y * elementsInChunk + threadIdx.x] = 0.0;
        }

        // Run calculations on shared memory matrix
        __syncthreads();
        for (int i = 0; i < blockDim.x; i++) {
            sum += As[i * elementsInChunk + threadIdx.y] * Bs[i * elementsInChunk + threadIdx.x];
        }
        __syncthreads();
    }
    C[y*outputSizeX + x] = sum;
}

__global__
void kMeanX(float* a, int aX, int aY, float* b)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < aX) {
        float sum = 0.0;
        for (int i = 0; i < aY; i++) {
            sum += a[i*aX + col];
        }
        b[col] = sum / aY;
    }
}

extern "C" void kAdd_kernel(dim3 d1, dim3 d2, float *a, float *b, int N) {
	    kAdd<<<d1, d2>>>(a, b, N);
}
extern "C" void kSubtract1D_kernel(dim3 d1, dim3 d2, float *a, float *b, int N) {
	    kSubtract1D<<<d1, d2>>>(a, b, N);
}
extern "C" void kAdd1D_kernel(dim3 d1, dim3 d2, float *a, float *b, int sizeX, int sizeY) {
	    kAdd1D<<<d1, d2>>>(a, b, sizeX, sizeY);
}
extern "C" void kScale1D_kernel(dim3 d1, dim3 d2, float *a, float factor, int N) {
	    kScale1D<<<d1, d2>>>(a, factor, N);
}

extern "C" void kAdd2D_kernel(dim3 d1, dim3 d2, float *a, float *b, int sizeX, int sizeY) {
	    kAdd2D<<<d1, d2>>>(a, b, sizeX, sizeY);
}

extern "C" void kSubtract2D_kernel(dim3 d1, dim3 d2, float *a, float *b, int sizeX, int sizeY) {
	    kSubtract2D<<<d1, d2>>>(a, b, sizeX, sizeY);
}

extern "C" void kScale2D_kernel(dim3 d1, dim3 d2, float *a, float factor, int sizeX, int sizeY) {
	    kScale2D<<<d1, d2>>>(a, factor, sizeX, sizeY);
}

extern "C" void kMultiply_kernel(dim3 d1, dim3 d2, int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
               float* A, int aX, int aY,
               float* B, int bX, int bY,
               float* C) {
	    kMultiply<<<d1, d2>>>(fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY, A, aX, aY, B, bX, bY, C);
}

extern "C" void kMultiplyWithSharedMemory_kernel(dim3 d1, dim3 d2, int d3, float* A, int aX, int aY,
                               float* B, int bX, int bY,
                               float* C) {
	    kMultiplyWithSharedMemory<<<d1, d2, d3>>>(A, aX, aY, B, bX, bY, C);
}

extern "C" void kMultiplyByTransposition_kernel(dim3 d1, dim3 d2, int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
                              float* A, int aX, int aY,
                              float* B, int bX, int bY,
                              float* C) {
	    kMultiplyByTransposition<<<d1, d2>>>(fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY, A, aX, aY, B, bX, bY, C);
}

extern "C" void kMultiplyByTranspositionWithSharedMemory_kernel(dim3 d1, dim3 d2, int d3, float* A, int aX, int aY,
                                              float* B, int bX, int bY,
                                              float* C) {
	    kMultiplyByTranspositionWithSharedMemory<<<d1, d2, d3>>>(A, aX, aY, B, bX, bY, C);
}

extern "C" void kTransposeAndMultiply_kernel(dim3 d1, dim3 d2, int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
                           float* A, int aX, int aY,
                           float* B, int bX, int bY,
                           float* C) {
	    kTransposeAndMultiply<<<d1, d2>>>(fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY, A, aX, aY, B, bX, bY, C);
}

extern "C" void kTransposeAndMultiplyWithSharedMemory_kernel(dim3 d1, dim3 d2, int d3, float* A, int aX, int aY,
                                           float* B, int bX, int bY,
                                           float* C) {
	    kTransposeAndMultiplyWithSharedMemory<<<d1, d2, d3>>>(A, aX, aY, B, bX, bY, C);
}

extern "C" void kMeanX_kernel(dim3 d1, dim3 d2, float* a, int aX, int aY, float* b) {
	    kMeanX<<<d1, d2>>>(a, aX, aY, b);
}