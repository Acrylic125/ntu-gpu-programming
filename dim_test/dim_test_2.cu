#include <cstdio>
#include <cuda_runtime.h>

__global__
void writeThread() {
	int tid = threadIdx.x
		+ threadIdx.y * blockDim.x
		+ threadIdx.z * blockDim.x * blockDim.y;

	printf(
		"tid=%d | tx=%d ty=%d tz=%d | bx=%d by=%d bz=%d\n",
		tid,
		threadIdx.x, threadIdx.y, threadIdx.z,
		blockDim.x, blockDim.y, blockDim.z
	);
}

int main() {
	int gDim = 4;
	int bDim = 512;
	writeThread<<<gDim, bDim>>>();

	cudaDeviceSynchronize();  
	printf("Done!\n");
	return 0;
}

