#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cstdlib>

__global__ void kernel_3d3d_arr(int* darr) {
	int block_threads = blockDim.x*blockDim.y*blockDim.z; // number of threads in a block

	int grid_z_offset = gridDim.x*gridDim.y*blockIdx.z*block_threads; // this gives the number of threads that are before the zth x-y block plane
	int grid_y_offset = gridDim.x*blockIdx.y*block_threads; // this gives the number of threads that are before the yth block row of the zth x-y block plane
	int grid_x_offset = blockIdx.x*block_threads; // this gives the number of threads that are before the xth block in the yth block row of the zth block x-y plane
	int block_offset = blockDim.x*blockDim.y*threadIdx.z; // this gives the number of threads that are before the zth x-y thread plane
	int row_offset = blockDim.x*threadIdx.y; // this gives the number of threads that are before the yth thread row of the zth x-y thread plane
	int tid = threadIdx.x; // this gives the number of threads that are before the xth thread of the yth thread row of the zth x-y thread plane

	int unique_id = grid_z_offset + grid_y_offset + grid_x_offset + block_offset + row_offset + tid;

	printf("Unique ID = %d\tArray element = %d\n", unique_id, darr[unique_id]);
}

int main() {
	int arr[64];
	
	for (int i = 0; i < 64; i++) {
		arr[i] = rand();
	}

	dim3 grid(2,2,2);
	dim3 block(2,2,2);

	int* darr; 
	// this is a pointer in the host memory that will point to the pointer in our device memory that will contain the addresses of our data in device memory

	cudaMalloc((void**)&darr, 64*sizeof(int)); // cudaMalloc takes a pointer to the pointer in the device memory as first argument. this is of type void

	cudaMemcpy(darr, arr, 64*sizeof(int), cudaMemcpyHostToDevice);

	kernel_3d3d_arr<<<grid, block>>>(darr);
	cudaDeviceSynchronize();

	cudaMemcpy(arr, darr, 64*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(darr);

	return 0;
}