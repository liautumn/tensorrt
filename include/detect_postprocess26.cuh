#ifndef POSTPROCESS26_CUH
#define POSTPROCESS26_CUH

#include <cuda_runtime_api.h>

namespace detect26 {
    const int NUM_BOX_ELEMENT = 8; // left, top, right, bottom, confidence, class, keepflag, row_index

    void decode_kernel_invoker(float *predict, int num_bboxes, int output_cdim,
                               float confidence_threshold,
                               float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                               cudaStream_t stream);
}

#endif //POSTPROCESS26_CUH
