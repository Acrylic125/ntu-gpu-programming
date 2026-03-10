#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
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
  
  // Only update interior points
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
        // Lambda2 since we assume dx = dy
        + lambda2 * (up + down + left + right - 4.0 * center);
  }
  
  // Boundary = 0 (Dirichlet)
  if (i < N && j < N && (i == 0 || i == N - 1 || j == 0 || j == N - 1))
  {
    next[i * N + j] = 0.0;
  }
}

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
  // std::string fname = "sim/sim_" + std::to_string(Lk) + ".txt";
  std::ofstream out(fname, std::ios::app);
  if (!firstSnapshot)
    out << "\n\n";
  for (int k = 0; k < gridPoints; k++)
    out << (k ? " " : "") << snapshot_vector[k];
  out << "\n";
  out.close();
}

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

// Global memory version
void qA1(bool saveSnapshots)
{
  if (saveSnapshots) {
    std::system("rm -rf sim/p1 2>/dev/null; mkdir -p sim/p1");
    std::system("rm -rf results/p1/with_save 2>/dev/null; mkdir -p results/p1/with_save");
  } else {
    std::system("rm -rf results/p1/without_save 2>/dev/null; mkdir -p results/p1/without_save");
  }
  int sizes[] = {
    1, 2, 4, 8
  };
  int blockSizes[] = {
    16, 32
  };
  double dx = 0.01;
  double dy = 0.01;
  double dt = 0.005;
  const int numSteps = 2000;

  double lambda = dt / dx; // c = 1, lambda = c dt / dx
  double lambda2 = lambda * lambda;

  for (int Lk : sizes) {
    // Multiply by 100, same as dividing by 0.01 but preserving the precision.
    int gridPointsOnAxis = Lk * 100;
    double grid[gridPointsOnAxis][gridPointsOnAxis];
    int gridPoints = gridPointsOnAxis * gridPointsOnAxis;
    std::cout << "Grid points: Lk: " << Lk << ", Grid Points: " << gridPoints << std::endl;
    // Initialise u(0, x, y) = sin(πx) sin(πy)
    for (int i = 0; i < gridPointsOnAxis; i++) {
      for (int j = 0; j < gridPointsOnAxis; j++) {
        // Check if boundary is correct
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

      // Start from grid[0][0]
      CUDA_CHECK(cudaMemcpy(d_u_curr, &grid[0][0], gridPoints * sizeof(double), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_u_prev, d_u_curr, gridPoints * sizeof(double), cudaMemcpyDeviceToDevice));

      dim3 blockSize(blockSizeDim, blockSizeDim);
      dim3 gridSize((N + blockSizeDim - 1) / blockSizeDim, (N + blockSizeDim - 1) / blockSizeDim);
      const int saveInterval = saveSnapshots ? (numSteps / 20) : 0;
      std::vector<double> snapshot_vector(gridPoints);
      std::string fname = "sim/p1/sim_" + std::to_string(Lk) + "_b" + std::to_string(blockSizeDim) + ".txt";

      cudaEvent_t start, stop;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));

      CUDA_CHECK(cudaDeviceSynchronize());

      for (int step = 0; step < numSteps; step++) {
        if (saveSnapshots && saveInterval > 0 && step % saveInterval == 0) {
          CUDA_CHECK(cudaDeviceSynchronize());
          save_snapshot(d_u_curr, snapshot_vector, gridPoints, Lk, (step == 0), fname);
        }
        if (step == 0)
          CUDA_CHECK(cudaEventRecord(start));
        wave2D_global<<<gridSize, blockSize>>>(d_u_prev, d_u_curr, d_u_next, N, lambda2);
        CUDA_CHECK(cudaGetLastError());
        // We will cycle these pointers to avoid memory allocation and deallocation.
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
      double updateThroughputGUpdates =
          (interiorPoints * numSteps) / runtimeSeconds / 1e9;
      double occupancyPercent =
          kernel_occupancy_percent((const void*)wave2D_global, blockSizeDim * blockSizeDim);
      std::cout << "Lk " << Lk << ", block " << blockSizeDim << "x" << blockSizeDim
                << ": " << kernelMs << " ms"
                << ", steps " << numSteps
                << ", bandwidth " << bandwidthGBs << " GB/s"
                << ", throughput " << updateThroughputGUpdates << " GUpdates/s"
                << ", occupancy " << occupancyPercent << "%" << std::endl;

      // Save final snapshot to results/p1/with_save or results/p1/without_save
      CUDA_CHECK(cudaDeviceSynchronize());
      std::string results_fname = saveSnapshots
          ? ("results/p1/with_save/sim_" + std::to_string(Lk) + "_b" + std::to_string(blockSizeDim) + ".txt")
          : ("results/p1/without_save/sim_" + std::to_string(Lk) + "_b" + std::to_string(blockSizeDim) + ".txt");
      save_snapshot(d_u_curr, snapshot_vector, gridPoints, Lk, true, results_fname);
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));

      CUDA_CHECK(cudaFree(d_u_prev));
      CUDA_CHECK(cudaFree(d_u_curr));
      CUDA_CHECK(cudaFree(d_u_next));
    }
  }
}

int main(int argc, char* argv[])
{
  std::cout << "Simulate wave evolution" << std::endl;
  bool saveSnapshots = (argc >= 2 && std::string(argv[1]) == "true");
  std::cout << "Should save snapshots: " << saveSnapshots << std::endl;
  qA1(saveSnapshots);
  return 0;
}
