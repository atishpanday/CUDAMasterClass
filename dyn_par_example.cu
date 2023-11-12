#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void dyn_par_kernel(int size, int depth) {
	printf("Depth = %d, tid = %d\n", depth, threadIdx.x);
	
	if(size == 1) {
		return;
	}
	
	if(threadIdx.x == 0) {
		dyn_par_kernel<<<1, size / 2>>>(size / 2, depth + 1);
	}
}

int main() {
	dyn_par_kernel<<<1, 16>>>(16, 0);
	cudaDeviceSynchronize();
	
	cudaDeviceReset();
	return 0;
}
