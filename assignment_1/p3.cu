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


// __global__ void q3_tensor_core_mul(
//   float x[MatrixSize_M][MatrixSize_K],
//   float y[MatrixSize_K][MatrixSize_N],
//   float result[MatrixSize_M][MatrixSize_N])
// {
//   wmma::fragment<wmma::matrix_a, 16, 16, 16, wmma::precision::tf32, wmma::row_major> a_frag;
//   wmma::fragment<wmma::matrix_b, 16, 16, 16, wmma::precision::tf32, wmma::row_major> b_frag;
//   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag, d_frag;

//   wmma::load_matrix_sync(a_frag, x, 16);
//   wmma::load_matrix_sync(b_frag, y, 16);
//   wmma::fill_fragment(c_frag, 0.0f);

//   wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

//   wmma::store_matrix_sync(result, d_frag, 16, wmma::row_major);
// }

void q3()
{
  // Rand
  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_real_distribution<float> dist(0.0f, 10.0f);

  float (*x)[MatrixSize_M];
  float (*y)[MatrixSize_K];
  float (*result)[MatrixSize_N];

  int M = MatrixSize_M;
  int N = MatrixSize_N;
  int K = MatrixSize_K;

  CUDA_CHECK(cudaMallocManaged(&x, M * K * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&y, K * N * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&result, M * N * sizeof(float)));

  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < K; j++)
    {
      x[i][j] = dist(engine);
    }
    for (int j = 0; j < N; j++)
    {
      result[i][j] = 0;
    }
  }
  for (int i = 0; i < K; i++)
  {
    for (int j = 0; j < N; j++)
    {
      y[i][j] = dist(engine);
    }
  }

  dim3 blockSizes2D[] = {
    dim3(4, 4),
    dim3(8, 8),
    dim3(16, 16),
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
