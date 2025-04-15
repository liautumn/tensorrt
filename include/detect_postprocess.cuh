#ifndef POSTPROCESS_CUH
#define POSTPROCESS_CUH
#include <vector>

namespace detect {
#define GPU_BLOCK_THREADS 1024
    const int NUM_BOX_ELEMENT = 8; // left, top, right, bottom, confidence, class,

    struct Box {
        float left, top, right, bottom, confidence;
        int class_label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int class_label)
            : left(left),
              top(top),
              right(right),
              bottom(bottom),
              confidence(confidence),
              class_label(class_label) {
        }
    };

    typedef std::vector<Box> BoxArray;

    void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                               float confidence_threshold, float nms_threshold,
                               float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                               cudaStream_t stream);
}

#endif //POSTPROCESS_CUH
