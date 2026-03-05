// Compile with: nvcc -o p2 p2.cu -lcusparse
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

#define CUSPARSE_CHECK(call)                                                \
  do {                                                                      \
    cusparseStatus_t status = (call);                                       \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                \
      fprintf(stderr, "cuSPARSE Error: %d (at: %s:%d)\n", (int)status, __FILE__, __LINE__); \
      exit((int)status);                                                    \
    }                                                                       \
  } while (0)

// Kernel: u_next = 2*u_curr - u_prev + lambda2 * (L*u_curr), then enforce Dirichlet boundary.
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

// Build 2D Laplacian in CSR: (L*u)[i,j] = u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1] - 4*u[i,j].
// Interior rows have 5 nonzeros; boundary rows have 0 nonzeros (so (L*u)[boundary]=0).
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

// cuSPARSE-based wave solver (sparse Laplacian + cusparseSpMV)
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

  cusparseHandle_t cusparseHandle = nullptr;
  CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

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
    double *d_u_prev = nullptr, *d_u_curr = nullptr, *d_u_next = nullptr, *d_Lu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u_prev, gridPoints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_curr, gridPoints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_next, gridPoints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Lu, gridPoints * sizeof(double)));

    // Start from grid[0][0]
    CUDA_CHECK(cudaMemcpy(d_u_curr, &grid[0][0], gridPoints * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_prev, d_u_curr, gridPoints * sizeof(double), cudaMemcpyDeviceToDevice));

    // Build sparse Laplacian in CSR on host
    std::vector<int> rowPtr, colIdx;
    std::vector<double> values;
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
    CUSPARSE_CHECK(cusparseCreateCsr(
      &matL,
      (int64_t)N * N, (int64_t)N * N, (int64_t)nnz,
      d_rowPtr, d_colIdx, d_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_64F
    ));

    cusparseDnVecDescr_t vec_curr = nullptr, vec_Lu = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_curr, (int64_t)gridPoints, d_u_curr, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_Lu, (int64_t)gridPoints, d_Lu, CUDA_R_64F));

    double one = 1.0, zero = 0.0;
    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
      cusparseHandle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, matL, vec_curr, &zero, vec_Lu,
      CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
    ));
    void *d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));

    int _blockSize[] = { 32, 32 };
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

      CUSPARSE_CHECK(cusparseDnVecSetValues(vec_curr, d_u_curr));
      CUSPARSE_CHECK(cusparseSpMV(
        cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, matL, vec_curr, &zero, vec_Lu,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer
      ));
      wave2D_update_from_Laplacian<<<gridSize, blockSize>>>(
        d_u_prev, d_u_curr, d_Lu, d_u_next, N, lambda2
      );
      CUDA_CHECK(cudaGetLastError());

      double *tmp = d_u_prev;
      d_u_prev = d_u_curr;
      d_u_curr = d_u_next;
      d_u_next = tmp;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelMs = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));
    std::cout << "Lk " << Lk << ": " << kernelMs << " ms (cuSPARSE SpMV)" << std::endl;

    // Save final snapshot to results/with_save or results/without_save
    CUDA_CHECK(cudaDeviceSynchronize());
    std::string results_fname = saveSnapshots
        ? ("results/with_save/sim_" + std::to_string(Lk) + ".txt")
        : ("results/without_save/sim_" + std::to_string(Lk) + ".txt");
    save_snapshot(d_u_curr, snapshot_vector, gridPoints, Lk, true, results_fname);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_curr));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_Lu));
    CUSPARSE_CHECK(cusparseDestroySpMat(matL));
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_Lu));
    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u_curr));
    CUDA_CHECK(cudaFree(d_u_next));
  }

  CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));
}

int main(int argc, char* argv[])
{
  std::cout << "Simulate wave evolution" << std::endl;
  bool saveSnapshots = (argc >= 2 && std::string(argv[1]) == "true");
  std::cout << "Should save snapshots: " << saveSnapshots << std::endl;
  qA1(saveSnapshots);
  return 0;
}
