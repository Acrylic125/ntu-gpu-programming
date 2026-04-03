#include "pipeline.cuh"
#include <cmath>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian kernel weights (normalised by 273).
// Stored as a constant array in GPU constant memory for fast broadcast reads.
// ─────────────────────────────────────────────────────────────────────────────
__constant__ float c_gauss[5][5] = {
    { 1.f/273,  4.f/273,  7.f/273,  4.f/273,  1.f/273 },
    { 4.f/273, 16.f/273, 26.f/273, 16.f/273,  4.f/273 },
    { 7.f/273, 26.f/273, 41.f/273, 26.f/273,  7.f/273 },
    { 4.f/273, 16.f/273, 26.f/273, 16.f/273,  4.f/273 },
    { 1.f/273,  4.f/273,  7.f/273,  4.f/273,  1.f/273 },
};


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 1 — Gaussian Blur (shared memory tiling with halo cells)
// ═════════════════════════════════════════════════════════════════════════════
//
// Background:
//   Each output pixel is a weighted average of its 5x5 neighbourhood.
//   Neighbouring output pixels share input pixels, so loading the input tile
//   into shared memory reduces global memory traffic significantly.
//   The shared tile must be larger than the output tile by GAUSS_RADIUS pixels
//   on every side — these extra pixels are called "halo cells".
//
// Shared memory layout:
//
//   ┌────────────────────────────┐  ← (TILE_W + 2*GAUSS_RADIUS) wide
//   │  halo  │  halo   │  halo   │  } GAUSS_RADIUS rows of halo
//   ├────────┼─────────┼─────────┤
//   │  halo  │ OUTPUT  │  halo   │  } TILE_H rows of output pixels
//   ├────────┼─────────┼─────────┤
//   │  halo  │  halo   │  halo   │  } GAUSS_RADIUS rows of halo
//   └────────────────────────────┘
//
// Your tasks:
//   1. Declare shared memory with the correct halo-extended dimensions.
//   2. Map each thread to a global (x, y) position.
//   3. Load the centre pixels AND halo pixels into shared memory cooperatively.
//      (Some threads may need to load more than one pixel.)
//   4. __syncthreads() before any computation.
//   5. Apply the 5x5 convolution from shared memory for in-bounds threads.
//   6. Write the result to `out`.
//
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gaussianBlurKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height)
{
    // Shared memory tile dimensions (centre + halo on each side).
    const int SMEM_W = TILE_W + 2 * GAUSS_RADIUS;
    const int SMEM_H = TILE_H + 2 * GAUSS_RADIUS;

    // TODO: Declare shared memory array of size SMEM_H x SMEM_W.
    __shared__ uint8_t tile[SMEM_H][SMEM_W];

    // TODO: Compute the global (x, y) output pixel this thread is
    //            responsible for.
    const int x = blockIdx.x * TILE_W + threadIdx.x;
    const int y = blockIdx.y * TILE_H + threadIdx.y;

    // TODO: Load shared memory cooperatively.
    //
    //   The input tile starts at global coordinates:
    //     tile_start_x = blockIdx.x * TILE_W - GAUSS_RADIUS
    //     tile_start_y = blockIdx.y * TILE_H - GAUSS_RADIUS
    //
    //   Each thread should load at least its own pixel at (threadIdx.x + GAUSS_RADIUS,
    //   threadIdx.y + GAUSS_RADIUS) in shared memory, plus any halo pixels it is
    //   responsible for. common strategy: iterate over the SMEM_H x SMEM_W region
    //   using a strided loop over the linearised thread index.
    //
    //   Boundary condition: clamp out-of-bounds global coordinates to [0, width-1]
    //   and [0, height-1] before indexing into `in`.

    const int tile_start_x = blockIdx.x * TILE_W - GAUSS_RADIUS; 
    const int tile_start_y = blockIdx.y * TILE_H - GAUSS_RADIUS; 

    const int tid = threadIdx.y * TILE_W + threadIdx.x;
    const int smem_size = SMEM_W * SMEM_H;

    for (int i = tid; i < smem_size; i += TILE_W * TILE_H) {
        const int sy = i / SMEM_W;
        const int sx = i % SMEM_W;

        // Clamp global coordinates to image bounds (clamp-to-edge).
        const int gx = max(0, min(tile_start_x + sx, width  - 1));
        const int gy = max(0, min(tile_start_y + sy, height - 1));

        tile[sy][sx] = in[gy * width + gx];
    }

    // TODO: Synchronise all threads before computing the convolution.
    __syncthreads();

    // TODO: Apply the 5x5 Gaussian convolution from shared memory.
    //
    //   Each thread computes one output pixel.
    //     out_x = blockIdx.x * TILE_W + threadIdx.x;
    //     out_y = blockIdx.y * TILE_H + threadIdx.y;
    //   Only threads whose (out_x, out_y) is within [0, width) x [0, height)
    //   should write to `out`.
    //
    //   Sum over ki = 0..4, kj = 0..4:
    //     sum += c_gauss[ki][kj] * smem[threadIdx.y + ki][threadIdx.x + kj]
    //
    //   Clamp (use roundf) the result to [0, 255] and cast to uint8_t before storing
    //   i.e., (uint8_t)min(max((int)roundf(sum), 0), 255);

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ki = 0; ki < 5; ki++) {
            for (int kj = 0; kj < 5; kj++) {
                sum += c_gauss[ki][kj] * tile[threadIdx.y + ki][threadIdx.x + kj];
            }
        }
        out[y * width + x] = (uint8_t)min(max((int)roundf(sum), 0), 255);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 2 — Sobel Edge Detection
// ═════════════════════════════════════════════════════════════════════════════
//
// Background
//   Two 3x3 kernels (Gx, Gy) measure intensity gradient in x and y directions.
//   Gradient magnitude = sqrt(Gx^2 + Gy^2), clamped to [0, 255].
//
//   Gx = [[-1, 0, 1],     Gy = [[ 1,  2,  1],
//         [-2, 0, 2],           [ 0,  0,  0],
//         [-1, 0, 1]]           [-1, -2, -1]]
//
// Both Gx and Gy must be computed in this single kernel.
// Shared memory tiling is optional but encouraged.
// Use clamp-to-edge for boundary pixels.
//
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sobelKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height)
{
    // TODO: For each pixel, read the 3x3 neighbourhood from `in`.
    //   Clamp out-of-bounds accesses to the nearest valid pixel.
    //   apply Gx and Gy convolutions, compute sqrt(Gx^2 + Gy^2),
    //   clamp to [0, 255], and write to `out`.

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    auto px = [&](int px, int py) -> float {
        px = min(max(px, 0), width - 1);
        py = min(max(py, 0), height - 1);
        return (float)in[py * width + px];
    };

    // 3x3 neighbourhood
    float p00 = px(x-1,y-1), p10 = px(x,y-1), p20 = px(x+1,y-1);
    float p01 = px(x-1,y  ), p11 = px(x,y  ), p21 = px(x+1,y  );
    float p02 = px(x-1,y+1), p12 = px(x,y+1), p22 = px(x+1,y+1);

    // Sobel X and Y responses.
    float gx = -p00 + p20 - 2 * p01 + 2 * p21 - p02 + p22;
    float gy = p00 + 2 * p10 + p20 - p02 - 2 * p12 - p22;

    float squared = gx * gx + gy * gy;
    float mag = sqrtf(squared);
    out[y * width + x] = (uint8_t)min(max((int)mag, 0), 255);
    // out[y * width + x] = (uint8_t)min(max((int)roundf(mag), 0), 255);
}


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3A — Histogram Kernel
// ═════════════════════════════════════════════════════════════════════════════
//
// Background:
//   Count how many pixels have each intensity value (0–255).
//   Many threads will try to increment the same bin simultaneously,
//   so atomic operations are required.
//
// `hist` is a device array of 256 unsigned ints, zero-initialised before launch.
//
// Optimisation hint (optional, but worth attempting):
//   Use a per-block shared memory histogram (256 unsigned ints), accumulate
//   locally with __atomicAdd on shared memory, then flush to global memory
//   once per block. This reduces contention on the 256 global counters.
//
// ─────────────────────────────────────────────────────────────────────────────
__global__ void histogramKernel(
    const uint8_t*  __restrict__ in,
    unsigned int*   hist,
    int width, int height)
{
    // TODO: Read the pixel value and atomically increment hist[pixel_value].
    const int total = width * height;
    const int linear_tid =
        (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
        (threadIdx.y * blockDim.x + threadIdx.x);
    const int stride = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    for (int i = linear_tid; i < total; i += stride) {
        atomicAdd(&hist[in[i]], 1u);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3B — CDF on host (solution given in multigpu.cu)
// ═════════════════════════════════════════════════════════════════════════════

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3C — Equalisation Kernel
// ═════════════════════════════════════════════════════════════════════════════
//
// Background:
//   Remap each pixel using:
//     new_val = round((CDF[old_val] - cdf_min) / (W*H - cdf_min) * 255)
//
// `cdf` is a device array of 256 floats from thrust::exclusive_scan, so:
//  cdf[i] = number of pixels with intensity STRICTLY LESS THAN i, cdf[0] = 0.
//  cdf_min is the first non-zero value in cdf[], found on the host after the scan.
//
// ─────────────────────────────────────────────────────────────────────────────
__global__ void equalizeKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    const float*   __restrict__ cdf,
    float          cdf_min,
    int width, int height)
{
    const size_t total = (size_t)width * height;

    const size_t tid =
        (size_t)(blockIdx.y * gridDim.x + blockIdx.x) *
        (blockDim.x * blockDim.y) +
        (threadIdx.y * blockDim.x + threadIdx.x);

    const size_t stride =
        (size_t)gridDim.x * gridDim.y *
        blockDim.x * blockDim.y;

    const float denom = (float)total - cdf_min;

    for (size_t i = tid; i < total; i += stride) {
        const uint8_t val = in[i];

        float mapped = 0.0f;

        if (denom > 0.0f) {
            mapped = (cdf[val] - cdf_min) / denom;
            mapped = mapped * 255.0f;
        }

        // Round and clamp
        int out_val = (int) roundf(mapped);  
        out_val = max(0, min(255, out_val));

        out[i] = (uint8_t)out_val;
    }
}
// __global__ void equalizeKernel(
//     const uint8_t* __restrict__ in,
//     uint8_t*       __restrict__ out,
//     const float*   cdf,
//     float          cdf_min,
//     int width, int height)
// {
//     // TODO: Read the input pixel, apply the equalisation formula,
//     //   clamp to [0, 255], cast to uint8_t, and write to `out`.
//     //   Total pixels = width * height.
//     const int total = width * height;
//     const int linear_tid =
//         (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
//         (threadIdx.y * blockDim.x + threadIdx.x);
//     const int stride = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
//     const float denom = (float)total - cdf_min;

//     for (int i = linear_tid; i < total; i += stride) {
//         const uint8_t old_val = in[i];
//         float mapped = 0.0f;
//         if (denom > 0.0f) {
//             mapped = ((cdf[(int)old_val] - cdf_min) / denom) * 255.0f;
//         }

//         int out_val = (int)roundf(mapped);
//         out_val = out_val < 0 ? 0 : (out_val > 255 ? 255 : out_val);
//         out[i] = (uint8_t)out_val;
//     }
// }
