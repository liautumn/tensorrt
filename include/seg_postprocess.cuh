//
// Created by autumn on 2025/4/18.
//

#ifndef SEG_POSTPROCESS_CUH
#define SEG_POSTPROCESS_CUH
#include <memory>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

namespace seg {
#define GPU_BLOCK_THREADS 1024
    using namespace std;

    const int NUM_BOX_ELEMENT = 8; // left, top, right, bottom, confidence, class,
    // keepflag, row_index(output)

    struct InstanceSegmentMap {
        int width = 0, height = 0; // width % 8 == 0
        int left = 0, top = 0; // 160x160 feature map
        unsigned char *data = nullptr; // is width * height memory

        InstanceSegmentMap(int width, int height);

        virtual ~InstanceSegmentMap();
    };

    struct Box {
        float left, top, right, bottom, confidence;
        int class_label;
        shared_ptr<InstanceSegmentMap> seg; // mask

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

    typedef vector<Box> BoxArray;

    __device__ __host__ void affine_project(float *matrix, float x, float y, float *ox, float *oy);

    void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                            int mask_width, int mask_height, unsigned char *mask_out,
                            int mask_dim, int out_width, int out_height, cudaStream_t stream);

    void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                               float confidence_threshold, float nms_threshold,
                               float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                               cudaStream_t stream);
}

#endif //SEG_POSTPROCESS_CUH
