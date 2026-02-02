#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <random>

#define CUDA_CHECK(call)                                    \
  do                                                        \
  {                                                         \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess)                                 \
    {                                                       \
      fprintf(stderr, "CUDA Error: %s (at: %s:%d)",         \
              cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(err);                                            \
    }                                                       \
  } while (0)

// Problem 1. Vector Addition (2 Marks)
__global__ void q1_add(float *x, float *y, float *sum)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  sum[i] = x[i] + y[i];
}

void q1()
{
  // Rand
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_real_distribution<float> dist(0.0f, 10.0f);

  int N = 1 << 30;

  float *x, *y, *sum;
  CUDA_CHECK(cudaMallocManaged(&x, N * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&y, N * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&sum, N * sizeof(float)));

  for (int i = 0; i < N; i++)
  {
    x[i] = dist(engine);
    y[i] = dist(engine);
    sum[i] = 0;
  }

  // block threads sizes to test, 32, 64, 128, 256
  int blockSizes[] = {32, 64, 128, 256};
  for (int bs : blockSizes)
  {
    int gridSize = (N + bs - 1) / bs;

    // Create start and stop events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    q1_add<<<gridSize, bs>>>(x, y, sum);
    // CUDA_CHECK(());
    // CUDA_CHECK(q1_add<<<gridSize, bs>>>(x, y, sum));

    // Measure elapsed time
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Block Size: " << bs << ", Time: " << elapsedTime << " ms" << std::endl;
  }
}

int main()
{
  q1();
  return 0;
}
