#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cstdlib>

__global__ void transpose_matrix(int* mat, int* trans, const int nx, const int ny) {

	int gid_x = blockDim.x * blockIdx.x + threadIdx.x;
	int gid_y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(gid_x < nx && gid_y < ny) {
		trans[gid_x * ny + gid_y] = mat[gid_y * nx + gid_x];
	}
}

int main() {

	const int nx = 256;
	const int ny = 256;
	const int size = nx * ny;
	int block_x = 128, block_y = 8;
	int h_mat[size], h_trans[size];
	
	// h_mat = (int*) malloc(size * sizeof(int));
	// h_trans = (int*) malloc(size * sizeof(int));
	
	for(int i = 0; i < size; i++) {
		h_mat[i] = 1 + rand() % 10;
	}
	 
	int* d_mat, * d_trans;
	
	cudaMalloc((void**) &d_mat, size * sizeof(int));
	cudaMalloc((void**) &d_trans, size * sizeof(int));
	
	cudaMemcpy(d_mat, h_mat, size * sizeof(int), cudaMemcpyHostToDevice);
	
	dim3 grid(nx / block_x, ny / block_y);
	dim3 block(block_x, block_y);
	
	transpose_matrix<<<grid, block>>>(d_mat, d_trans, nx, ny);
	cudaDeviceSynchronize();
	
	cudaMemcpy(h_trans, d_trans, size * sizeof(int), cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < 10; i++) {
		for(int j = 0; j < 10; j++) {
			printf("%d ", h_mat[i * nx + j]);
		}
		printf("\n");
	}
	
	printf("\n\n");
	
	for(int i = 0; i < 10; i++) {
		for(int j = 0; j < 10; j++) {
			printf("%d ", h_trans[i * nx + j]);
		}
		printf("\n");
	}
	
	cudaFree(d_mat);
	cudaFree(d_trans);
	
	cudaDeviceReset();
	
	return 0;
}
