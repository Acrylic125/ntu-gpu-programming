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

// Problem 2. Matrix Addition (3 Marks)
// 1d grid and block
__global__ void q2_add_1d(
  float x[MatrixSize_N][MatrixSize_M],
   float y[MatrixSize_N][MatrixSize_M],
    float sum[MatrixSize_N][MatrixSize_M])
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  int row = i / MatrixSize_M;
  int col = i % MatrixSize_M;

  sum[row][col] = x[row][col] + y[row][col];
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
  std::uniform_real_distribution<float> dist(0.0f, 100.0f);

  int M = MatrixSize_M;
  int N = MatrixSize_N;

  float (*x)[MatrixSize_M], (*h_x)[MatrixSize_M];
  float (*y)[MatrixSize_M], (*h_y)[MatrixSize_M];
  float (*sum)[MatrixSize_M], (*h_sum)[MatrixSize_M];

  size_t arrSize = N * M * sizeof(float);
  h_x = (float (*)[MatrixSize_M])malloc(arrSize);
  h_y = (float (*)[MatrixSize_M])malloc(arrSize);
  h_sum = (float (*)[MatrixSize_M])malloc(arrSize);

  for (int i = 0; i < MatrixSize_N; i++)
  {
    for (int j = 0; j < MatrixSize_M; j++)
    {
      h_x[i][j] = dist(engine);
      h_y[i][j] = dist(engine);
      h_sum[i][j] = 0;
    }
  }

  CUDA_CHECK(cudaMalloc((void**) &x, arrSize));
  CUDA_CHECK(cudaMalloc((void**) &y, arrSize));
  CUDA_CHECK(cudaMalloc((void**) &sum, arrSize));
  CUDA_CHECK(cudaMemcpy(x, h_x, arrSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(y, h_y, arrSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(sum, h_sum, arrSize, cudaMemcpyHostToDevice));

  std::cout << "1D Grid and Block:" << std::endl;
  // When we convert it to 2D, we will consider the same number
  // of total threads.
  int blockSizes[] = {
    16,
    64, 
    256, 
    1024
  };
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
    long long _flops = (long long)M * N; // One add operation per element
    double flops = (seconds > 0.0f) ? ((double)_flops / seconds) : 0.0;
    std::cout << "Block Size: " << bs << ", Time: " << elapsedTime << " ms, FLOPs: " << flops << std::endl;

    // Copy sum back to host
    CUDA_CHECK(cudaMemcpy(h_sum, sum, arrSize, cudaMemcpyDeviceToHost));
    if (!verifyResultMatrix(h_x, h_y, h_sum))
      std::cout << "Verification failed!" << std::endl;
  }

  // 2D Grid and Block
  std::cout << "2D Grid and Block:" << std::endl;
  // Equiv. number of threads in 1D.
  dim3 blockSizes2D[] = {
    // Duplicate to prove theres some initial overhead.
    dim3(4, 4),
    dim3(4, 4),
    dim3(16, 1),
    dim3(1, 16),
    dim3(8, 8),
    dim3(64, 1),
    dim3(1, 64),
    dim3(16, 16),
    dim3(256, 1),
    dim3(1, 256),
    dim3(32, 32),
    dim3(1024, 1),
    dim3(1, 1024)
  };
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
    long long _flops = (long long)N * M; // One add operation per element
    double flops = (seconds > 0.0f) ? ((double)_flops / seconds) : 0.0;
    std::cout << "Block Size: (" << bs.x << ", " << bs.y << "), Time: " << elapsedTime << " ms, FLOPs: " << flops << std::endl;

    // Copy sum back to host
    CUDA_CHECK(cudaMemcpy(h_sum, sum, arrSize, cudaMemcpyDeviceToHost));
    if (!verifyResultMatrix(h_x, h_y, h_sum))
    {
      std::cout << "Verification failed!" << std::endl;
    }
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
