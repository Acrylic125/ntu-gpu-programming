#include "multigpu.cuh"
#include "pipeline.cuh"
#include "pgm_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

// ─────────────────────────────────────────────────────────────────────────────
// process_batch_on_device
//
// Runs the full three-stage pipeline on a sub-batch of images using a single
// GPU (identified by device_id). Each image is processed sequentially on its
// own CUDA stream so that H→D transfers, kernels, and D→H transfers can
// overlap across images.
//
// You do not need to modify the stream structure here — but read it carefully,
// as you will need to describe it in your report.
// ─────────────────────────────────────────────────────────────────────────────
static void process_batch_on_device(std::vector<ImageEntry>& sub_batch, int device_id)
{
    // TODO: Call cudaSetDevice(device_id) BEFORE any CUDA API call.

    int W = sub_batch[0].width;
    int H = sub_batch[0].height;
    size_t img_bytes = (size_t)W * H * sizeof(uint8_t);
    int    n_images  = (int)sub_batch.size();

    // ── Per-image device buffers ──────────────────────────────────────────
    // Each image needs its own set of device buffers so streams can operate
    // independently. Allocate them all up front.
    // TODO: Allocate per-image device buffers.


    // ── Per-image CUDA streams ────────────────────────────────────────────
    // One stream per image: all operations for image i are serialised on
    // stream[i], but different images' streams run concurrently on the GPU.

    // TODO: Create n_images CUDA streams.
    //   std::vector<cudaStream_t> streams(n_images);
    //   for (int i = 0; i < n_images; i++) cudaStreamCreate(&streams[i]);


    // ── Submit all images to the GPU ──────────────────────────────────────
    for (int i = 0; i < n_images; i++) {

        // TODO: cudaMemcpyAsync host→device for image i on streams[i].


        // TODO: Stage 1 Launch gaussianBlurKernel on streams[i].
        //   Grid: ceil(W/TILE_W) x ceil(H/TILE_H) blocks, TILE_W x TILE_H threads.

        // TODO: Stage 2 Launch sobelKernel on streams[i].

        // TODO: Zero-initialise d_hist[i] with cudaMemsetAsync on streams[i].

        // Stage 3 

        // TODO: Stage 3A: Launch histogramKernel on streams[i].
        
        // TODO: Stage 3B (you can directly use the code provided here or implement your own)
        //   CDF via thrust (thrust uses default stream — must sync first)
        //   Then compute the CDF (thrust calls execute on CPU):
        //     thrust::device_ptr<unsigned int> hist_ptr(d_hist[i]);
        //     thrust::device_ptr<float>        cdf_ptr(d_cdf[i]);
        //     thrust::exclusive_scan(hist_ptr, hist_ptr + 256, cdf_ptr);
        //   Find cdf_min on the host.
        //   Copy d_cdf[i] back to a host array (synchronously is fine — it is
        //   only 256 floats), find the first non-zero entry, and pass it to
        //   equalizeKernel below.
        //     float h_cdf[256];
        //     cudaMemcpy(h_cdf, d_cdf[i], cdf_bytes, cudaMemcpyDeviceToHost);
        //     float cdf_min = 0.f;
        //     for (int b = 0; b < 256; b++) {
        //         if (h_cdf[b] > 0.f) { cdf_min = h_cdf[b]; break; }
        //      }

        // TODO: Stage 3C Launch equalizeKernel on streams[i].

        // TODO: cudaMemcpyAsync device→host for the equalised output on streams[i].
    }

    // ── Wait for all images to finish ─────────────────────────────────────
    // TODO: cudaStreamSynchronize each stream.


    // ── Save results ──────────────────────────────────────────────────────
    for (int i = 0; i < n_images; i++) {
        pgm_save(sub_batch[i].output_path, sub_batch[i].host_out, W, H);
    }

    // ── Clean up ──────────────────────────────────────────────────────────
    // TODO: Destroy all streams and free all device buffers.

}


// ─────────────────────────────────────────────────────────────────────────────
// run_pipeline_multigpu  (public entry point)
// ─────────────────────────────────────────────────────────────────────────────
void run_pipeline_multigpu(std::vector<ImageEntry>& batch)
{
    // TODO: Detect the number of available GPUs.
    int num_gpus = 0;

    if (num_gpus < 2) {
        // TODO: Print a warning and fall back to GPU 0.
        // process_batch_on_device(batch, 0);
        return;
    }

    // TODO: Split `batch` into two sub-batches.
    //   GPU 0 gets the first N/2 images; GPU 1 gets the remainder.

    // TODO: Process sub0 on GPU 0, then sub1 on GPU 1.
    //   process_batch_on_device(sub0, 0);
    //   process_batch_on_device(sub1, 1);
}

void run_pipeline_singlegpu(std::vector<ImageEntry>& batch)
{
    process_batch_on_device(batch, 0);
}
