#ifndef OBBPOSTPROCESS_CUH
#define OBBPOSTPROCESS_CUH
#include <vector>

namespace obb {
#define GPU_BLOCK_THREADS 1024
    using namespace std;
    const int NUM_BOX_ELEMENT = 8; // cx, cy, w, h, angle, confidence, class_label, keepflag

    struct Box {
        float center_x, center_y, width, height, angle, confidence;
        int class_label;

        Box() = default;

        Box(float center_x, float center_y, float width, float height, float angle, float confidence, int class_label)
            : center_x(center_x), center_y(center_y), width(width), height(height), angle(angle),
              confidence(confidence), class_label(class_label) {
        }
    };

    typedef vector<Box> BoxArray;

    void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                   float confidence_threshold, float nms_threshold,
                                   float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                                   cudaStream_t stream);
}

#endif //OBBPOSTPROCESS_CUH
