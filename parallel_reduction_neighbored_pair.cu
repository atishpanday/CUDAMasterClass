#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cstdlib>
#include<iostream>

__global__ void neighbored_pair_reduction(int* input, int* temp, int size) {
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size) {
		return;
	}

	for (int offset = 1; offset < blockDim.x; offset *= 2) {
		if (tid % (2 * offset) == 0) {
			input[gid] += input[gid + offset];
		}
		__syncthreads();
	}

	if (tid == 0) {
		temp[blockIdx.x] = input[gid];
	}
}

int main() {
	const int N = 1 << 14;
	const int block_size = 128;
	const int num_blocks = N / block_size;
	int arr[N];
	int h_temp[num_blocks];

	for (int i = 0; i < N; i++) {
		arr[i] = rand() * 10 / RAND_MAX;
	}

	for (int i = 1; i < N; i++) {
		arr[0] += arr[i];
	}

	int* d_arr, * d_temp;

	dim3 block(block_size);
	dim3 grid(num_blocks);

	cudaMalloc((void**) &d_arr, N*sizeof(int));
	cudaMalloc((void**) &d_temp, num_blocks*sizeof(int));

	cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(d_temp, 0, num_blocks * sizeof(int));

	neighbored_pair_reduction<<<grid, block>>>(d_arr, d_temp, N);
	cudaDeviceSynchronize();

	cudaMemcpy(h_temp, d_temp, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 1; i < num_blocks; i++) {
		h_temp[0] += h_temp[i];
	}

	std::cout << "CPU results: "<< arr[0] << ", GPU results : " << h_temp[0];

	cudaFree(d_arr);
	cudaFree(d_temp);
	cudaDeviceReset();

	return 0;
}