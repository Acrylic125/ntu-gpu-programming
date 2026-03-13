// Combined submission: p1 (qA1), p1b (qA2), p2 (q2)
// Compile with: nvcc -o submission submission.cu -lcusparse

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <math.h>
#include <cuda_runtime.h>
#include <cusparse.h>
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

#define CUSPARSE_CHECK(call)                                                \
  do {                                                                      \
    cusparseStatus_t status = (call);                                       \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                \
      fprintf(stderr, "cuSPARSE Error: %d (at: %s:%d)\n", (int)status, __FILE__, __LINE__); \
      exit((int)status);                                                    \
    }                                                                       \
  } while (0)

// Part 1 with global memory, no shared memory optimizations
__global__
void wave2D_global(
  const double* prev,
  const double* cur,
  double* next,
  int N,
  double lambda2
)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i > 0 && i < N - 1 && j > 0 && j < N - 1)
  {
    int idx = i * N + j;

    double center = cur[idx];

    double up    = cur[(i - 1) * N + j];
    double down  = cur[(i + 1) * N + j];
    double left  = cur[i * N + (j - 1)];
    double right = cur[i * N + (j + 1)];

    next[idx] =
        2.0 * center
        - prev[idx]
        + lambda2 * (up + down + left + right - 4.0 * center);
  }

  if (i < N && j < N && (i == 0 || i == N - 1 || j == 0 || j == N - 1))
  {
    next[i * N + j] = 0.0;
  }
}

// Part 1b with shared memory optimizations
__global__
void wave2D_shared(
  const double* prev,
  const double* cur,
  double* next,
  int N,
  double lambda2
)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tilePitch = blockDim.x + 2;

  extern __shared__ double tile[];

  if (i < N && j < N)
    tile[(ty + 1) * tilePitch + (tx + 1)] = cur[i * N + j];
  else
    tile[(ty + 1) * tilePitch + (tx + 1)] = 0.0;

  // Load halos in x-direction
  if (tx == 0) {
    int j_left = j - 1;
    tile[(ty + 1) * tilePitch] = (i >= 0 && i < N && j_left >= 0)
        ? cur[i * N + j_left]
        : 0.0;
  }
  if (tx == blockDim.x - 1) {
    int j_right = j + 1;
    tile[(ty + 1) * tilePitch + (blockDim.x + 1)] = (i >= 0 && i < N && j_right < N)
        ? cur[i * N + j_right]
        : 0.0;
  }

  // Load halos in y-direction
  if (ty == 0) {
    int i_up = i - 1;
    tile[tx + 1] = (i_up >= 0 && j >= 0 && j < N)
        ? cur[i_up * N + j]
        : 0.0;
  }
  if (ty == blockDim.y - 1) {
    int i_down = i + 1;
    tile[(blockDim.y + 1) * tilePitch + (tx + 1)] = (i_down < N && j >= 0 && j < N)
        ? cur[i_down * N + j]
        : 0.0;
  }

  // All threads must wait for the slowest thread to load their slice of the tile.
  __syncthreads();

  if (i > 0 && i < N - 1 && j > 0 && j < N - 1)
  {
    const int idx = i * N + j;

    double center = tile[(ty + 1) * tilePitch + (tx + 1)];
    double up     = tile[ty * tilePitch + (tx + 1)];
    double down   = tile[(ty + 2) * tilePitch + (tx + 1)];
    double left   = tile[(ty + 1) * tilePitch + tx];
    double right  = tile[(ty + 1) * tilePitch + (tx + 2)];

    next[idx] =
        2.0 * center
        - prev[idx]
        + lambda2 * (up + down + left + right - 4.0 * center);
  }

  if (i < N && j < N && (i == 0 || i == N - 1 || j == 0 || j == N - 1))
  {
    next[i * N + j] = 0.0;
  }
}

// Part 1b with shared memory optimizations and single tile load
// Alternative implementation to reduce warp divergence.
__global__
void wave2D_shared_with_single_tile_load(
  const double* prev,
  const double* cur,
  double* next,
  int N,
  double lambda2
)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tilePitch = blockDim.x + 2;
  const int tileHeight = blockDim.y + 2;
  const int tileElements = tilePitch * tileHeight;

  extern __shared__ double tile[];

  if (tx == 0) {
    for (int linearIdx = ty; linearIdx < tileElements; linearIdx += blockDim.y) {
      int localRow = linearIdx / tilePitch;
      int localCol = linearIdx % tilePitch;
      int globalI = blockIdx.y * blockDim.y + localRow - 1;
      int globalJ = blockIdx.x * blockDim.x + localCol - 1;

      tile[linearIdx] = (globalI >= 0 && globalI < N && globalJ >= 0 && globalJ < N)
          ? cur[globalI * N + globalJ]
          : 0.0;
    }
  }

  __syncthreads();

  if (i > 0 && i < N - 1 && j > 0 && j < N - 1)
  {
    const int idx = i * N + j;

    double center = tile[(ty + 1) * tilePitch + (tx + 1)];
    double up     = tile[ty * tilePitch + (tx + 1)];
    double down   = tile[(ty + 2) * tilePitch + (tx + 1)];
    double left   = tile[(ty + 1) * tilePitch + tx];
    double right  = tile[(ty + 1) * tilePitch + (tx + 2)];

    next[idx] =
        2.0 * center
        - prev[idx]
        + lambda2 * (up + down + left + right - 4.0 * center);
  }

  if (i < N && j < N && (i == 0 || i == N - 1 || j == 0 || j == N - 1))
  {
    next[i * N + j] = 0.0;
  }
}

// Part 2 with cuSPARSE for Laplacian matrix
__global__
void wave2D_update_from_Laplacian(
  const double* prev,
  const double* cur,
  const double* Lu,
  double* next,
  int N,
  double lambda2
)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i * N + j;
  if (i >= N || j >= N) return;
  if (i > 0 && i < N - 1 && j > 0 && j < N - 1)
    next[idx] = 2.0 * cur[idx] - prev[idx] + lambda2 * Lu[idx];
  else
    next[idx] = 0.0;
}

static void build_Laplacian_CSR(
  int N,
  std::vector<int>& rowPtr,
  std::vector<int>& colIdx,
  std::vector<double>& values
)
{
  const int n = N * N;
  rowPtr.resize(n + 1);
  int nnz = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int idx = i * N + j;
      rowPtr[idx] = nnz;
      if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        colIdx.push_back((i - 1) * N + j);
        values.push_back(1.0);
        colIdx.push_back((i + 1) * N + j);
        values.push_back(1.0);
        colIdx.push_back(i * N + (j - 1));
        values.push_back(1.0);
        colIdx.push_back(i * N + (j + 1));
        values.push_back(1.0);
        colIdx.push_back(i * N + j);
        values.push_back(-4.0);
        nnz += 5;
      }
    }
  }
  rowPtr[n] = nnz;
}

// Helpers, only used for submission.
static void save_snapshot(
  const double* d_u_curr,
  std::vector<double>& snapshot_vector,
  int gridPoints,
  int Lk,
  bool firstSnapshot,
  std::string fname
)
{
  CUDA_CHECK(cudaMemcpy(snapshot_vector.data(), d_u_curr, gridPoints * sizeof(double), cudaMemcpyDeviceToHost));
  std::ofstream out(fname, std::ios::app);
  if (!firstSnapshot)
    out << "\n\n";
  for (int k = 0; k < gridPoints; k++)
    out << (k ? " " : "") << snapshot_vector[k];
  out << "\n";
  out.close();
}

// Theoretical occupancy percentage calculation. 
// Does not factor in warp divergence.
static double kernel_occupancy_percent(const void* kernel, int threadsPerBlock, size_t dynamicSmemBytes = 0)
{
  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int activeBlocksPerSm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &activeBlocksPerSm, kernel, threadsPerBlock, dynamicSmemBytes));

  return 100.0 * activeBlocksPerSm * threadsPerBlock / prop.maxThreadsPerMultiProcessor;
}

void qA1(bool saveSnapshots)
{
  // Clear artifacts from previous runs.
  if (saveSnapshots) {
    std::system("rm -rf sim/p1 2>/dev/null; mkdir -p sim/p1");
    std::system("rm -rf results/p1/with_save 2>/dev/null; mkdir -p results/p1/with_save");
  } else {
    std::system("rm -rf results/p1/without_save 2>/dev/null; mkdir -p results/p1/without_save");
  }

  // Lk values
  int sizes[] = {
    1, 2, 4, 8
  };
  // Block sizes to test (For verification only)
  int blockSizes[][2] = {
    {4, 4},
    {8, 8},
    {16, 16},
    {32, 32}
  };
  double dx = 0.01;
  double dy = 0.01;
  double dt = 0.005;
  const int numSteps = 2000;

  double lambda = dt / dx;
  double lambda2 = lambda * lambda;

  for (int Lk : sizes) {
    // Multiply by 100, same as dividing by 0.01 but preserving the precision.
    int gridPointsOnAxis = Lk * 100;
    
    // Initialize simulation grid
    double grid[gridPointsOnAxis][gridPointsOnAxis];
    int gridPoints = gridPointsOnAxis * gridPointsOnAxis;
    std::cout << "[qA1] Grid points: Lk: " << Lk << ", Grid Points: " << gridPoints << std::endl;
    for (int i = 0; i < gridPointsOnAxis; i++) {
      for (int j = 0; j < gridPointsOnAxis; j++) {
        if (i == 0 || i == gridPointsOnAxis - 1 || j == 0 || j == gridPointsOnAxis - 1) {
          grid[i][j] = 0;
        } else {
          grid[i][j] = sin(M_PI * i * dx) * sin(M_PI * j * dy);
        }
      }
    }

    const int N = gridPointsOnAxis;
    const long long interiorPoints = 1LL * (N - 2) * (N - 2);
    const double bytesPerGridUpdate = 6.0 * sizeof(double);
    for (const auto& blockSizePair : blockSizes) {
      int blockSizeX = blockSizePair[0];
      int blockSizeY = blockSizePair[1];
      double *d_u_prev = nullptr, *d_u_curr = nullptr, *d_u_next = nullptr;
      CUDA_CHECK(cudaMalloc(&d_u_prev, gridPoints * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&d_u_curr, gridPoints * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&d_u_next, gridPoints * sizeof(double)));

      CUDA_CHECK(cudaMemcpy(d_u_curr, &grid[0][0], gridPoints * sizeof(double), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_u_prev, d_u_curr, gridPoints * sizeof(double), cudaMemcpyDeviceToDevice));

      dim3 blockSize(blockSizeX, blockSizeY);
      dim3 gridSize((N + blockSizeX - 1) / blockSizeX, (N + blockSizeY - 1) / blockSizeY);
      const int saveInterval = saveSnapshots ? (numSteps / 20) : 0;
      std::vector<double> snapshot_vector(gridPoints);
      std::string fname = "sim/p1/sim_" + std::to_string(Lk) + "_b" +
          std::to_string(blockSizeX) + "x" + std::to_string(blockSizeY) + ".txt";

      cudaEvent_t start, stop;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));

      CUDA_CHECK(cudaDeviceSynchronize());

      for (int step = 0; step < numSteps; step++) {
        // Save snapshot every 20 steps, if enabled.
        // When benchmarking, we will not save snapshots since this is overhead.
        if (saveSnapshots && saveInterval > 0 && step % saveInterval == 0) {
          CUDA_CHECK(cudaDeviceSynchronize());
          save_snapshot(d_u_curr, snapshot_vector, gridPoints, Lk, (step == 0), fname);
        }
        // Start timing when step == 0, to avoid overhead of first step.
        if (step == 0)
          CUDA_CHECK(cudaEventRecord(start));
        wave2D_global<<<gridSize, blockSize>>>(d_u_prev, d_u_curr, d_u_next, N, lambda2);
        CUDA_CHECK(cudaGetLastError());

        // Cycle pointers to avoid memory allocation and deallocation.
        double *tmp = d_u_prev;
        d_u_prev = d_u_curr;
        d_u_curr = d_u_next;
        d_u_next = tmp;
      }
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));

      float kernelMs = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));
      double runtimeSeconds = kernelMs / 1e3;
      double totalBytesTransferred = interiorPoints * numSteps * bytesPerGridUpdate;
      double bandwidthGBs = totalBytesTransferred / runtimeSeconds / 1e9;
      double updatesPerSecond =
          (interiorPoints * numSteps) / runtimeSeconds;
      double occupancyPercent =
          kernel_occupancy_percent((const void*)wave2D_global, blockSizeX * blockSizeY);
      std::cout << "Lk " << Lk << ", block " << blockSizeX << "x" << blockSizeY
                << ": " << kernelMs << " ms"
                << ", bandwidth " << bandwidthGBs << " GB/s"
                << ", throughput " << updatesPerSecond << " updates/s"
                << ", occupancy " << occupancyPercent << "%" << std::endl;

      CUDA_CHECK(cudaDeviceSynchronize());

      // Save final result artifact.
      std::string results_fname = saveSnapshots
          ? ("results/p1/with_save/sim_" + std::to_string(Lk) + "_b" +
             std::to_string(blockSizeX) + "x" + std::to_string(blockSizeY) + ".txt")
          : ("results/p1/without_save/sim_" + std::to_string(Lk) + "_b" +
             std::to_string(blockSizeX) + "x" + std::to_string(blockSizeY) + ".txt");
      save_snapshot(d_u_curr, snapshot_vector, gridPoints, Lk, true, results_fname);
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));

      CUDA_CHECK(cudaFree(d_u_prev));
      CUDA_CHECK(cudaFree(d_u_curr));
      CUDA_CHECK(cudaFree(d_u_next));
    }
  }
}

void qA2(bool saveSnapshots)
{
  // Clear artifacts from previous runs.
  if (saveSnapshots) {
    std::system("rm -rf sim/p1b 2>/dev/null; mkdir -p sim/p1b");
    std::system("rm -rf results/p1b/with_save 2>/dev/null; mkdir -p results/p1b/with_save");
  } else {
    std::system("rm -rf results/p1b/without_save 2>/dev/null; mkdir -p results/p1b/without_save");
  }

  int sizes[] = {
    1, 2, 4, 8
  };
  // Block sizes to test (For verification only)
  // Using what we got from A1, we will stick to 16.
  int blockSizes[] = {
    16, 32
  };
  double dx = 0.01;
  double dy = 0.01;
  double dt = 0.005;
  const int numSteps = 2000;

  double lambda = dt / dx;
  double lambda2 = lambda * lambda;

  for (int Lk : sizes) {
    // Multiply by 100, same as dividing by 0.01 but preserving the precision.
    int gridPointsOnAxis = Lk * 100;

    // Initialize simulation grid
    double grid[gridPointsOnAxis][gridPointsOnAxis];
    int gridPoints = gridPointsOnAxis * gridPointsOnAxis;
    std::cout << "[qA2] Grid points: Lk: " << Lk << ", Grid Points: " << gridPoints << std::endl;
    for (int i = 0; i < gridPointsOnAxis; i++) {
      for (int j = 0; j < gridPointsOnAxis; j++) {
        if (i == 0 || i == gridPointsOnAxis - 1 || j == 0 || j == gridPointsOnAxis - 1) {
          grid[i][j] = 0;
        } else {
          grid[i][j] = sin(M_PI * i * dx) * sin(M_PI * j * dy);
        }
      }
    }

    const int N = gridPointsOnAxis;
    const long long interiorPoints = 1LL * (N - 2) * (N - 2);
    const double bytesPerGridUpdate = 6.0 * sizeof(double);
    for (int blockSizeDim : blockSizes) {
      double *d_u_prev = nullptr, *d_u_curr = nullptr, *d_u_next = nullptr;
      CUDA_CHECK(cudaMalloc(&d_u_prev, gridPoints * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&d_u_curr, gridPoints * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&d_u_next, gridPoints * sizeof(double)));

      CUDA_CHECK(cudaMemcpy(d_u_curr, &grid[0][0], gridPoints * sizeof(double), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_u_prev, d_u_curr, gridPoints * sizeof(double), cudaMemcpyDeviceToDevice));

      dim3 blockSize(blockSizeDim, blockSizeDim);
      dim3 gridSize((N + blockSizeDim - 1) / blockSizeDim, (N + blockSizeDim - 1) / blockSizeDim);
      const int saveInterval = saveSnapshots ? (numSteps / 20) : 0;
      std::vector<double> snapshot_vector(gridPoints);
      std::string fname = "sim/p1b/sim_" + std::to_string(Lk) + "_b" + std::to_string(blockSizeDim) + ".txt";

      cudaEvent_t start, stop;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));

      CUDA_CHECK(cudaDeviceSynchronize());

      for (int step = 0; step < numSteps; step++) {
        // Save snapshot every 20 steps, if enabled.
        // When benchmarking, we will not save snapshots since this is overhead.
        if (saveSnapshots && saveInterval > 0 && step % saveInterval == 0) {
          CUDA_CHECK(cudaDeviceSynchronize());
          save_snapshot(d_u_curr, snapshot_vector, gridPoints, Lk, (step == 0), fname);
        }
        // Start timing when step == 0, to avoid overhead of first step.
        if (step == 0)
          CUDA_CHECK(cudaEventRecord(start));
        size_t sharedBytes = (blockSizeDim + 2) * (blockSizeDim + 2) * sizeof(double);
        wave2D_shared_with_single_tile_load<<<gridSize, blockSize, sharedBytes>>>(
            d_u_prev, d_u_curr, d_u_next, N, lambda2);
        CUDA_CHECK(cudaGetLastError());

        // Cycle pointers to avoid memory allocation and deallocation.
        double *tmp = d_u_prev;
        d_u_prev = d_u_curr;
        d_u_curr = d_u_next;
        d_u_next = tmp;
      }
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));

      float kernelMs = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));
      size_t sharedBytes = (blockSizeDim + 2) * (blockSizeDim + 2) * sizeof(double);
      double runtimeSeconds = kernelMs / 1e3;
      double totalBytesTransferred = interiorPoints * numSteps * bytesPerGridUpdate;
      double bandwidthGBs = totalBytesTransferred / runtimeSeconds / 1e9;
      double updatesPerSecond =
          (interiorPoints * numSteps) / runtimeSeconds;
      double occupancyPercent =
          kernel_occupancy_percent((const void*)wave2D_shared_with_single_tile_load,
                                   blockSizeDim * blockSizeDim, sharedBytes);
      std::cout << "Lk " << Lk << ", block " << blockSizeDim << "x" << blockSizeDim
                << ": " << kernelMs << " ms"
                << ", bandwidth " << bandwidthGBs << " GB/s"
                << ", throughput " << updatesPerSecond << " updates/s"
                << ", occupancy " << occupancyPercent << "%" << std::endl;

      CUDA_CHECK(cudaDeviceSynchronize());

      // Save final result artifact.
      std::string results_fname = saveSnapshots
          ? ("results/p1b/with_save/sim_" + std::to_string(Lk) + "_b" + std::to_string(blockSizeDim) + ".txt")
          : ("results/p1b/without_save/sim_" + std::to_string(Lk) + "_b" + std::to_string(blockSizeDim) + ".txt");
      save_snapshot(d_u_curr, snapshot_vector, gridPoints, Lk, true, results_fname);
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));

      CUDA_CHECK(cudaFree(d_u_prev));
      CUDA_CHECK(cudaFree(d_u_curr));
      CUDA_CHECK(cudaFree(d_u_next));
    }
  }
}

void q2(bool saveSnapshots)
{
  // Clear artifacts from previous runs.
  if (saveSnapshots) {
    std::system("rm -rf sim/p2 2>/dev/null; mkdir -p sim/p2");
    std::system("rm -rf results/p2/with_save 2>/dev/null; mkdir -p results/p2/with_save");
  } else {
    std::system("rm -rf results/p2/without_save 2>/dev/null; mkdir -p results/p2/without_save");
  }

  // Lk values
  int sizes[] = {
    1, 2, 4, 8
  };
  // Block sizes to test, only used by update kernel.
  // Again, using what we got from A1, we will stick to 16.
  int blockSizes[] = {
    16, 32
  };
  double dx = 0.01;
  double dy = 0.01;
  double dt = 0.005;
  const int numSteps = 2000;

  double lambda = dt / dx;
  double lambda2 = lambda * lambda;

  cusparseHandle_t cusparseHandle = nullptr;
  CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

  for (int Lk : sizes) {
    // Multiply by 100, same as dividing by 0.01 but preserving the precision.
    int gridPointsOnAxis = Lk * 100;

    // Initialize simulation grid
    double grid[gridPointsOnAxis][gridPointsOnAxis];
    int gridPoints = gridPointsOnAxis * gridPointsOnAxis;
    std::cout << "[q2] Grid points: Lk: " << Lk << ", Grid Points: " << gridPoints << std::endl;
    for (int i = 0; i < gridPointsOnAxis; i++) {
      for (int j = 0; j < gridPointsOnAxis; j++) {
        if (i == 0 || i == gridPointsOnAxis - 1 || j == 0 || j == gridPointsOnAxis - 1) {
          grid[i][j] = 0;
        } else {
          grid[i][j] = sin(M_PI * i * dx) * sin(M_PI * j * dy);
        }
      }
    }

    const int N = gridPointsOnAxis;
    const long long interiorPoints = 1LL * (N - 2) * (N - 2);
    const double bytesPerGridUpdate = 6.0 * sizeof(double);
    std::vector<int> rowPtr, colIdx;
    std::vector<double> values;
    // Build sparse Laplacian in CSR on host.
    // Example: for N = 4, the interior point (1, 1) has flat index 5.
    // Its CSR row stores colIdx = [1, 9, 4, 6, 5], values = [1, 1, 1, 1, -4].
    build_Laplacian_CSR(N, rowPtr, colIdx, values);
    const int nnz = (int)values.size();

    int *d_rowPtr = nullptr, *d_colIdx = nullptr;
    double *d_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (N * N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(), (N * N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t matL = nullptr;
    // Just another way to represent the sparse matrix for us to use in cuSPARSE.
    CUSPARSE_CHECK(cusparseCreateCsr(
      &matL,
      (int64_t)N * N, (int64_t)N * N, (int64_t)nnz,
      d_rowPtr, d_colIdx, d_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_64F
    ));

    for (int blockSizeDim : blockSizes) {
      double *d_u_prev = nullptr, *d_u_curr = nullptr, *d_u_next = nullptr, *d_Lu = nullptr;
      CUDA_CHECK(cudaMalloc(&d_u_prev, gridPoints * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&d_u_curr, gridPoints * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&d_u_next, gridPoints * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&d_Lu, gridPoints * sizeof(double)));

      CUDA_CHECK(cudaMemcpy(d_u_curr, &grid[0][0], gridPoints * sizeof(double), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_u_prev, d_u_curr, gridPoints * sizeof(double), cudaMemcpyDeviceToDevice));

      cusparseDnVecDescr_t vec_curr = nullptr, vec_Lu = nullptr;
      CUSPARSE_CHECK(cusparseCreateDnVec(&vec_curr, (int64_t)gridPoints, d_u_curr, CUDA_R_64F));
      CUSPARSE_CHECK(cusparseCreateDnVec(&vec_Lu, (int64_t)gridPoints, d_Lu, CUDA_R_64F));

      double one = 1.0, zero = 0.0;
      size_t bufferSize = 0;
      // Example: before multiplying a 16 x 16 sparse matrix by a 16-entry
      // vector, ask cuSPARSE for the exact temporary buffer size it needs.
      CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, matL, vec_curr, &zero, vec_Lu,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
      ));
      void *d_buffer = nullptr;
      CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));

      dim3 blockSize(blockSizeDim, blockSizeDim);
      dim3 gridSize((N + blockSizeDim - 1) / blockSizeDim, (N + blockSizeDim - 1) / blockSizeDim);
      const int saveInterval = saveSnapshots ? (numSteps / 20) : 0;
      std::vector<double> snapshot_vector(gridPoints);
      std::string fname = "sim/p2/sim_" + std::to_string(Lk) + "_b" + std::to_string(blockSizeDim) + ".txt";

      cudaEvent_t start, stop;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));

      CUDA_CHECK(cudaDeviceSynchronize());

      for (int step = 0; step < numSteps; step++) {
        // Save snapshot every 20 steps, if enabled.
        // When benchmarking, we will not save snapshots since this is overhead.
        if (saveSnapshots && saveInterval > 0 && step % saveInterval == 0) {
          CUDA_CHECK(cudaDeviceSynchronize());
          save_snapshot(d_u_curr, snapshot_vector, gridPoints, Lk, (step == 0), fname);
        }
        // Start timing when step == 0, to avoid overhead of first step.
        if (step == 0)
          CUDA_CHECK(cudaEventRecord(start));

        CUSPARSE_CHECK(cusparseDnVecSetValues(vec_curr, d_u_curr));
        // Performs Lu = L * u_curr.
        CUSPARSE_CHECK(cusparseSpMV(
          cusparseHandle,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &one, matL, vec_curr, &zero, vec_Lu,
          CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer
        ));
        // Use Lu to update the next wave field.
        wave2D_update_from_Laplacian<<<gridSize, blockSize>>>(
          d_u_prev, d_u_curr, d_Lu, d_u_next, N, lambda2
        );
        CUDA_CHECK(cudaGetLastError());

        // Cycle pointers to avoid memory allocation and deallocation.
        double *tmp = d_u_prev;
        d_u_prev = d_u_curr;
        d_u_curr = d_u_next;
        d_u_next = tmp;
      }
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));

      float kernelMs = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));
      double runtimeSeconds = kernelMs / 1e3;
      double totalBytesTransferred = interiorPoints * numSteps * bytesPerGridUpdate;
      double bandwidthGBs = totalBytesTransferred / runtimeSeconds / 1e9;
      double updatesPerSecond =
          (interiorPoints * numSteps) / runtimeSeconds;
      double occupancyPercent =
          kernel_occupancy_percent((const void*)wave2D_update_from_Laplacian,
                                   blockSizeDim * blockSizeDim);
      std::cout << "Lk " << Lk << ", block " << blockSizeDim << "x" << blockSizeDim
                << ": " << kernelMs << " ms"
                << ", bandwidth " << bandwidthGBs << " GB/s"
                << ", throughput " << updatesPerSecond << " updates/s"
                << ", occupancy " << occupancyPercent << "%" << std::endl;

      CUDA_CHECK(cudaDeviceSynchronize());

      // Save final result artifact.
      std::string results_fname = saveSnapshots
          ? ("results/p2/with_save/sim_" + std::to_string(Lk) + "_b" + std::to_string(blockSizeDim) + ".txt")
          : ("results/p2/without_save/sim_" + std::to_string(Lk) + "_b" + std::to_string(blockSizeDim) + ".txt");
      save_snapshot(d_u_curr, snapshot_vector, gridPoints, Lk, true, results_fname);
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));

      CUSPARSE_CHECK(cusparseDestroyDnVec(vec_curr));
      CUSPARSE_CHECK(cusparseDestroyDnVec(vec_Lu));
      CUDA_CHECK(cudaFree(d_buffer));
      CUDA_CHECK(cudaFree(d_Lu));
      CUDA_CHECK(cudaFree(d_u_prev));
      CUDA_CHECK(cudaFree(d_u_curr));
      CUDA_CHECK(cudaFree(d_u_next));
    }

    CUSPARSE_CHECK(cusparseDestroySpMat(matL));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_values));
  }

  CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));
}

int main(int argc, char* argv[])
{
  bool saveSnapshots = (argc >= 2 && std::string(argv[1]) == "true");

  qA1(saveSnapshots);
  qA2(saveSnapshots);
  q2(saveSnapshots);
  return 0;
}
