#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void hello() {
	int tid = threadIdx.x;
	if(tid == 0) {
		printf("Hello ");
	}
	else {
		printf("World ");
	}
}

int main () {
	hello<<<1, 2>>>();
	cudaDeviceSynchronize();
	return 0;
}
