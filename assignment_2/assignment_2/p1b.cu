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

// Tile size (must match launch configuration)
#define BLOCK_X 32
#define BLOCK_Y 32

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
void wave2D_shared(
  const double* prev,
  const double* cur,
  double* next,
  int N,
  double lambda2
)
{
  // Global indices
  const int j = blockIdx.x * blockDim.x + threadIdx.x; 
  const int i = blockIdx.y * blockDim.y + threadIdx.y; 

  // Local indices in shared tile (offset by 1 for halo)
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  __shared__ double tile[BLOCK_Y + 2][BLOCK_X + 2];

  // Load center point
  if (i < N && j < N)
    tile[ty + 1][tx + 1] = cur[i * N + j];
  else
    tile[ty + 1][tx + 1] = 0.0;

  // Load halos in x-direction
  if (tx == 0) {
    int j_left = j - 1;
    tile[ty + 1][0] = (i >= 0 && i < N && j_left >= 0)
        ? cur[i * N + j_left]
        : 0.0;
  }
  if (tx == blockDim.x - 1) {
    int j_right = j + 1;
    tile[ty + 1][BLOCK_X + 1] = (i >= 0 && i < N && j_right < N)
        ? cur[i * N + j_right]
        : 0.0;
  }

  // Load halos in y-direction
  if (ty == 0) {
    int i_up = i - 1;
    tile[0][tx + 1] = (i_up >= 0 && j >= 0 && j < N)
        ? cur[i_up * N + j]
        : 0.0;
  }
  if (ty == blockDim.y - 1) {
    int i_down = i + 1;
    tile[BLOCK_Y + 1][tx + 1] = (i_down < N && j >= 0 && j < N)
        ? cur[i_down * N + j]
        : 0.0;
  }

  // Corners of the tile
  if (tx == 0 && ty == 0) {
    int i_up = i - 1;
    int j_left = j - 1;
    tile[0][0] = (i_up >= 0 && j_left >= 0)
        ? cur[i_up * N + j_left]
        : 0.0;
  }
  if (tx == blockDim.x - 1 && ty == 0) {
    int i_up = i - 1;
    int j_right = j + 1;
    tile[0][BLOCK_X + 1] = (i_up >= 0 && j_right < N)
        ? cur[i_up * N + j_right]
        : 0.0;
  }
  if (tx == 0 && ty == blockDim.y - 1) {
    int i_down = i + 1;
    int j_left = j - 1;
    tile[BLOCK_Y + 1][0] = (i_down < N && j_left >= 0)
        ? cur[i_down * N + j_left]
        : 0.0;
  }
  if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
    int i_down = i + 1;
    int j_right = j + 1;
    tile[BLOCK_Y + 1][BLOCK_X + 1] = (i_down < N && j_right < N)
        ? cur[i_down * N + j_right]
        : 0.0;
  }

  __syncthreads();

  // Only update interior points
  if (i > 0 && i < N - 1 && j > 0 && j < N - 1)
  {
    const int idx = i * N + j;

    double center = tile[ty + 1][tx + 1];
    double up     = tile[ty    ][tx + 1];
    double down   = tile[ty + 2][tx + 1];
    double left   = tile[ty + 1][tx    ];
    double right  = tile[ty + 1][tx + 2];

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

// Global memory version
void qA1(bool saveSnapshots)
{
  if (saveSnapshots) {
    std::system("rm -rf sim 2>/dev/null; mkdir -p sim");
    std::system("rm -rf results/with_save 2>/dev/null; mkdir -p results/with_save");
  } else {
    std::system("rm -rf results/without_save 2>/dev/null; mkdir -p results/without_save");
  }
  int sizes[] = {
    1, 2, 4, 8
  };
  double dx = 0.01;
  double dy = 0.01;
  double dt = 0.005;

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
    double *d_u_prev = nullptr, *d_u_curr = nullptr, *d_u_next = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u_prev, gridPoints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_curr, gridPoints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_next, gridPoints * sizeof(double)));

    // Start from grid[0][0]
    CUDA_CHECK(cudaMemcpy(d_u_curr, &grid[0][0], gridPoints * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_prev, d_u_curr, gridPoints * sizeof(double), cudaMemcpyDeviceToDevice));

    int _blockSize[] = {
      32, 32
    };
    dim3 blockSize(_blockSize[0], _blockSize[1]);
    dim3 gridSize((N + _blockSize[0] - 1) / _blockSize[0], (N + _blockSize[1] - 1) / _blockSize[1]);
    const int numSteps = (int)(1.0 / dt);
    const int saveInterval = saveSnapshots ? (numSteps / 20) : 0;
    std::vector<double> snapshot_vector(gridPoints);
    std::string fname = "sim/sim_" + std::to_string(Lk) + ".txt";

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
      wave2D_shared<<<gridSize, blockSize>>>(d_u_prev, d_u_curr, d_u_next, N, lambda2);
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
    std::cout << "Lk " << Lk << ": " << kernelMs << " ms" << std::endl;

    // Save final snapshot to results/with_save or results/without_save
    CUDA_CHECK(cudaDeviceSynchronize());
    std::string results_fname = saveSnapshots
        ? ("results/with_save/sim_" + std::to_string(Lk) + ".txt")
        : ("results/without_save/sim_" + std::to_string(Lk) + ".txt");
    save_snapshot(d_u_curr, snapshot_vector, gridPoints, Lk, true, results_fname);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u_curr));
    CUDA_CHECK(cudaFree(d_u_next));
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
