#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cstdlib>

__global__ void neighbored_pair_reduction(int* input, int* temp, int size) {
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size) {
		return;
	}

	for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {	// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] -> [1, 5, 9, 13, 17, 21, 25, 29]
		if (tid < offset) {
			input[gid] += input[gid + offset];
		}
		__syncthreads();
	}

	if (tid == 0) {
		temp[blockIdx.x] = input[gid];
	}
}

int cpu_sum(int arr[], int N) {
	int sum = 0;
	for (int i = 0; i < N; i++) {
		sum += arr[i];
	}
	return sum;
}

int main() {
	const int N = 1 << 14;
	const int block_size = 128;
	const int num_blocks = N / block_size;
	int arr[N];
	int h_temp[num_blocks];
	int cpu_result = 0;

	for (int i = 0; i < N; i++) {
		arr[i] = (int) 10.0 * (rand() / RAND_MAX + 1.0);
	}

	cpu_result = cpu_sum(arr, N);

	dim3 block(block_size);
	dim3 grid(num_blocks);

	int* d_arr, * d_temp;

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

	printf("Number of blocks: %d, CPU results: %d, GPU results : %d", num_blocks, cpu_result, h_temp[0]);

	cudaFree(d_arr);
	cudaFree(d_temp);
	cudaDeviceReset();

	return 0;
}
