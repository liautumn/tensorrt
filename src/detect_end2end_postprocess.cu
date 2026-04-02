#include "detect_end2end_postprocess.cuh"
#include <logger.h>

namespace detect_end2end {
    constexpr int GPU_BLOCK_THREADS_END2END = 1024;

    static __device__ void affine_project(float *matrix, float x, float y, float *ox,
                                          float *oy) {
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    static __global__ void decode_kernel(float *predict, int num_bboxes, int output_cdim,
                                         float confidence_threshold, float *invert_affine_matrix,
                                         float *parray, int MAX_IMAGE_BOXES) {
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= num_bboxes) return;

        float *pitem = predict + output_cdim * position;
        float confidence = pitem[4];
        if (confidence < confidence_threshold) return;

        int label = static_cast<int>(roundf(pitem[5]));
        if (label < 0) return;

        float left = pitem[0];
        float top = pitem[1];
        float right = pitem[2];
        float bottom = pitem[3];
        affine_project(invert_affine_matrix, left, top, &left, &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

        if (right <= left || bottom <= top) return;

        int index = atomicAdd(parray, 1);
        if (index >= MAX_IMAGE_BOXES) return;

        float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = label;
        *pout_item++ = 1; // end-to-end output has already removed duplicate boxes
        *pout_item++ = position;
    }

    static dim3 grid_dims(int num_jobs) {
        int num_block_threads = num_jobs < GPU_BLOCK_THREADS_END2END ? num_jobs : GPU_BLOCK_THREADS_END2END;
        return dim3((num_jobs + num_block_threads - 1) / (float) num_block_threads);
    }

    static dim3 block_dims(int num_jobs) {
        return num_jobs < GPU_BLOCK_THREADS_END2END ? num_jobs : GPU_BLOCK_THREADS_END2END;
    }

    void decode_kernel_invoker(float *predict, int num_bboxes, int output_cdim,
                               float confidence_threshold,
                               float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                               cudaStream_t stream) {
        auto grid = grid_dims(num_bboxes);
        auto block = block_dims(num_bboxes);
        checkKernel(decode_kernel<<<grid, block, 0, stream>>>(
                        predict, num_bboxes, output_cdim, confidence_threshold, invert_affine_matrix, parray,
                        MAX_IMAGE_BOXES));
    }
}
