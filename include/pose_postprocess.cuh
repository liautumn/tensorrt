#ifndef POSE_POSTPROCESS_CUH
#define POSE_POSTPROCESS_CUH
#include <vector>
#include <opencv2/core/types.hpp>

namespace pose {
#define GPU_BLOCK_THREADS 1024
    using namespace std;
    constexpr int NUM_KEYPOINTS = 17; // COCO Keypoins
    constexpr int NUM_BOX_ELEMENT = 6 + 3 * NUM_KEYPOINTS;

    struct Box {
        float left, top, right, bottom, confidence;
        vector<cv::Point3f> keypoints;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence)
            : left(left), top(top), right(right), bottom(bottom), confidence(confidence) {
            keypoints.reserve(NUM_KEYPOINTS);
        }
    };

    typedef vector<Box> BoxArray;

    void decode_kernel_invoker(float *predict, int num_bboxes, int output_cdim,
                               float confidence_threshold, float nms_threshold,
                               float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                               cudaStream_t stream);
}

#endif //POSE_POSTPROCESS_CUH
