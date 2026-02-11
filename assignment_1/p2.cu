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


#define MatrixSize_M 8192
#define MatrixSize_N 8192
#define MatrixSize_K 8192

// Problem 2. Matrix Addition (3 Marks)
// 1d grid and block
__global__ void q2_add_1d(float x[MatrixSize_N][MatrixSize_M], float y[MatrixSize_N][MatrixSize_M], float sum[MatrixSize_N][MatrixSize_M])
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  int total = MatrixSize_N * MatrixSize_M;
  if (i < total)
  {
    int row = i / MatrixSize_M;
    int col = i % MatrixSize_M;

    sum[row][col] = x[row][col] + y[row][col];
  }
}

// 2d grid and block
__global__ void q2_add_2d(float x[MatrixSize_N][MatrixSize_M], float y[MatrixSize_N][MatrixSize_M], float sum[MatrixSize_N][MatrixSize_M])
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  sum[i][j] = x[i][j] + y[i][j];
}

bool verifyResultMatrix(float x[MatrixSize_N][MatrixSize_M],
                        float y[MatrixSize_N][MatrixSize_M],
                        float sum[MatrixSize_N][MatrixSize_M])
{
  for (int i = 0; i < MatrixSize_N; ++i)
  {
    for (int j = 0; j < MatrixSize_M; ++j)
    {
      float expected = x[i][j] + y[i][j];
      if (fabs(sum[i][j] - expected) > 1e-5f)
      {
        return false;
      }
    }
  }
  return true;
}

void q2()
{
  // Rand
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_real_distribution<float> dist(0.0f, 10.0f);

  int M = MatrixSize_M;
  int N = MatrixSize_N;

  float (*x)[MatrixSize_M];
  float (*y)[MatrixSize_M];
  float (*sum)[MatrixSize_M];
  // float x[MatrixSize_N][MatrixSize_M], y[MatrixSize_N][MatrixSize_M], sum[MatrixSize_N][MatrixSize_M];
  // float *x, *y, *sum;
  CUDA_CHECK(cudaMallocManaged(&x, N * M * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&y, N * M * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&sum, N * M * sizeof(float)));

  for (int i = 0; i < MatrixSize_N; i++)
  {
    for (int j = 0; j < MatrixSize_M; j++)
    {
      x[i][j] = dist(engine);
      y[i][j] = dist(engine);
      sum[i][j] = 0;
    }
  }

  std::cout << "1D Grid and Block:" << std::endl;
  // When we convert it to 2D, we will consider the same number
  // of total threads.
  int blockSizes[] = {16, 64, 256, 1024};
  for (int bs : blockSizes)
  {
    int gridSize = ((N * M) + bs - 1) / bs;

    // Create start and stop events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    q2_add_1d<<<gridSize, bs>>>(x, y, sum);
    CUDA_CHECK(cudaGetLastError()); 

    // Measure elapsed time
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); 

    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    float seconds = elapsedTime / 1000.0f;
    long long flops = M * N; // One add operation per element
    double gFlops = (flops / seconds) / 1e9;
    std::cout << "Block Size: " << bs << ", Time: " << elapsedTime << " ms, FLOPs: " << gFlops << std::endl;
  }
  if (verifyResultMatrix(x, y, sum))
  {
    std::cout << "Verification passed!" << std::endl;
  }
  else
  {
    std::cout << "Verification failed!" << std::endl;
  }

  // 2D Grid and Block
  std::cout << "2D Grid and Block:" << std::endl;
  // Equiv. number of threads in 1D.
  dim3 blockSizes2D[] = {dim3(4, 4), dim3(8, 8), dim3(16, 16), dim3(32, 32)};
  for (dim3 bs : blockSizes2D)
  {
    dim3 gridSize((N + bs.x - 1) / bs.x, (M + bs.y - 1) / bs.y);

    // Create start and stop events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    q2_add_2d<<<gridSize, bs>>>(x, y, sum);
    CUDA_CHECK(cudaGetLastError()); 

    // Measure elapsed time
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); 

    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    float seconds = elapsedTime / 1000.0f;
    long long flops = N * M; // One add operation per element
    double gFlops = (flops / seconds) / 1e9;
    std::cout << "Block Size: (" << bs.x << ", " << bs.y << "), Time: " << elapsedTime << " ms, FLOPs: " << gFlops << std::endl;
  }
  if (verifyResultMatrix(x, y, sum))
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
  std::cout << "Running Problem 2: Matrix Addition" << std::endl;
  q2();
  return 0;
}
