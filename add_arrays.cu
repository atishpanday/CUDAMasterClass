#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cstdlib>
#include <time.h>

__global__ void dkernel(int* a, int* b, int* c, int size) {
	int tid = threadIdx.x;
	if (tid < size) {
		c[tid] = a[tid] + b[tid];
	}
	printf("A[%d] = %d, B[%d] = %d, C[%d] = %d\n", tid, a[tid], tid, b[tid], tid, c[tid]);
}

int main() {
	time_t t;
	srand((unsigned) time (&t));

	const int N = 10;
	int A[N], B[N], C[N] = {0};

	int* dA, * dB, * dC;

	for (int i = 0; i < N; i++) {
		A[i] = (int) 10 * (rand() / RAND_MAX + 1.0);
		B[i] = (int) 10 * (rand() / RAND_MAX + 1.0);
	}

	cudaMalloc((void**)&dA, N * sizeof(int));
	cudaMalloc((void**)&dB, N * sizeof(int));
	cudaMalloc((void**)&dC, N * sizeof(int));

	cudaMemcpy(dA, A, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, N * sizeof(int), cudaMemcpyHostToDevice);

	dkernel<<<1, 10>>>(dA, dB, dC, N);
	cudaDeviceSynchronize();

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	cudaDeviceReset();

	return 0;
}
