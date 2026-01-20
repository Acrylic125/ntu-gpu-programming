#include <iostream>
#include <limits>
#include <cmath>

bool areAboutEqual_abs(float a, float b, float epsilon = 0.1) {
	    return std::abs(a - b) <= epsilon;
}

__global__
void addVecs(float *x, float *y, float *sum) {
	int i = threadIdx.x;

	sum[i] = x[i] + y[i];
}

int main() {
	float *x, *y, *sum;

	int arrSize = 1 << 20;
	cudaMallocManaged(&x, arrSize * sizeof(float));
	cudaMallocManaged(&y, arrSize * sizeof(float));
	cudaMallocManaged(&sum, arrSize * sizeof(float));

	for (int i = 0; i < arrSize; i++) {
		x[i] = i;
		y[i] = 100;
	}

	addVecs<<<1, 256>>>(x, y, sum);

	for (int i = 0; i < arrSize; i++) {
		if (areAboutEqual_abs(sum[i], x[i] + 100, 0.1)) {
			std::cout << "FUCK " << sum[i] << " != " << (x[i] + 100) <<  " at " << i << std::endl;
		}
	}

	cudaFree(x);
	cudaFree(y);
	cudaFree(sum);

	std::cout << "Done!" << std::endl;

	return 0;
}
