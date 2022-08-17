
#include "cukernel.h"

#define N 100

__global__ void addKernel(int* c, int* a, int* b, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b[i];
	}
}

#include <stdio.h>

extern "C" int compute() {

	int *darra, *darrb, *darrc;
	int all_correct = 1;

	int *arra = (int*) malloc(sizeof(int) * (N * N));
	int *arrb = (int*) malloc(sizeof(int) * (N * N));
	int *arrc = (int*) malloc(sizeof(int) * (N * N));

	cudaMalloc((void**)&darra, (N * N) * sizeof(int));
	cudaMalloc((void**)&darrb, (N * N) * sizeof(int));
	cudaMalloc((void**)&darrc, (N * N) * sizeof(int));

	for (int i = 0;i < (N * N);i++) {
		arra[i] = i;
		arrb[i] = i;
	}

	cudaMemcpy(darra, arra, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(darrb, arrb, (N * N) * sizeof(int), cudaMemcpyHostToDevice);

	addKernel<<<N, N>>>(darrc, darra, darrb, N * N);

	cudaMemcpy(arrc, darrc, (N * N)*sizeof(int), cudaMemcpyDeviceToHost);


	for (int i = 0;i < (N * N);i++) {
		if (arrc[i] != arra[i] + arrb[i]) {
			all_correct = 0;
			// printf("r[%d] is incorrect\n", i);
		}
	}

	if (all_correct) {
		fprintf(stderr, "all r[i] is correct\n");
	} else {
		fprintf(stderr, "some r[i] are incorrect\n");
	}

	cudaFree(darra);
	cudaFree(darrb);
	cudaFree(darrc);

	return all_correct;
}
