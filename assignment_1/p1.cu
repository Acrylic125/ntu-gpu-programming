#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <random>
#include <mma.h>

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

bool verifyResult(float *a, float *b, float *c, int n)
{
  for (int i = 0; i < n; i++)
  {
    float expected = a[i] + b[i];
    if (fabs(c[i] - expected) > 1e-5)
    {
      printf("Verification failed at index %d: expected %f, got %f\n",
             i, expected, c[i]);
      return false;
    }
  }
  return true;
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
    CUDA_CHECK(cudaGetLastError()); 

    // Measure elapsed time
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); 

    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    float seconds = elapsedTime / 1000.0f;
    long long flops = N; // One add operation per element
    double gFlops = (flops / seconds) / 1e9;
    std::cout << "Block Size: " << bs << ", Time: " << elapsedTime << " ms, FLOPs: " << gFlops << std::endl;
  }
  if (verifyResult(x, y, sum, N))
  {
    std::cout << "Verification passed!" << std::endl;
  }
  else
  {
    std::cout << "Verification failed!" << std::endl;
  }

  // Free memory
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  CUDA_CHECK(cudaFree(sum));
}


int main()
{
  std::cout << "Running Problem 1: Vector Addition" << std::endl;
  q1();
  return 0;
}
