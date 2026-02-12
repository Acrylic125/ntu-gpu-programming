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




#define MatrixSize_M 8192
#define MatrixSize_N 8192
#define MatrixSize_K 8192


// Naive implementation of matrix multiplication
__global__ void q3_naive_mul(
  float x[MatrixSize_M][MatrixSize_K],
   float y[MatrixSize_K][MatrixSize_N], 
  float result[MatrixSize_M][MatrixSize_N])
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  result[i][j] = 0;
  for (int k = 0; k < MatrixSize_K; k++)
  {
    result[i][j] += x[i][k] * y[k][j];
  }
}

void q3()
{
  // Rand
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  int M = MatrixSize_M;
  int N = MatrixSize_N;
  int K = MatrixSize_K;
  size_t size_x = M * K * sizeof(float);
  size_t size_y = K * N * sizeof(float);
  size_t size_result = M * N * sizeof(float);

  float (*x)[MatrixSize_K], (*h_x)[MatrixSize_K];
  float (*y)[MatrixSize_N], (*h_y)[MatrixSize_N];
  float (*result)[MatrixSize_N], (*h_result)[MatrixSize_N];

  h_x = (float (*)[MatrixSize_K])malloc(size_x);
  h_y = (float (*)[MatrixSize_N])malloc(size_y);
  h_result = (float (*)[MatrixSize_N])malloc(size_result);

  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < K; j++)
    {
      h_x[i][j] = dist(engine);
    }
    for (int j = 0; j < N; j++)
    {
      h_result[i][j] = 0;
    }
  }
  for (int i = 0; i < K; i++)
  {
    for (int j = 0; j < N; j++)
    {
      h_y[i][j] = dist(engine);
    }
  }

  CUDA_CHECK(cudaMalloc((void**) &x, size_x));
  CUDA_CHECK(cudaMalloc((void**) &y, size_y));
  CUDA_CHECK(cudaMalloc((void**) &result, size_result));
  CUDA_CHECK(cudaMemcpy(x, h_x, size_x, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(y, h_y, size_y, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(result, h_result, size_result, cudaMemcpyHostToDevice));

  dim3 blockSizes2D[] = {
    // Duplicate to prove theres some initial overhead.
    dim3(4, 4),
    dim3(4, 4),
    // 32
    dim3(1, 32),
    dim3(4, 8),
    // 64
    dim3(8, 8),
    // 128
    dim3(16, 16),
    // 256
    dim3(32, 32),
  };
  for (dim3 bs : blockSizes2D)
  {
    // Since we are using 1 thread to compute each individual element, 
    // we have M x N threads in total to manage.
    dim3 gridSize((M + bs.x - 1) / bs.x, (N + bs.y - 1) / bs.y);

    // Create start and stop events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    q3_naive_mul<<<gridSize, bs>>>(x, y, result);
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError()); 

    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    float seconds = elapsedTime / 1000.0f;
    // One multiply and one add per element (2 FLOPs)
    long long _flops = (long long)M * N * K * 2LL; 
    double flops = (seconds > 0.0f) ? ((double)_flops / seconds) : 0.0;
    std::cout << "Block Size: (" << bs.x << ", " << bs.y << "), Time: " << elapsedTime << " ms, FLOPs: " << flops << std::endl;
  }
}

int main()
{
  std::cout << "Running Problem 3: Matrix Multiplication" << std::endl;
  q3();
  return 0;
}
