#ifndef POSTPROCESS_CUH
#define POSTPROCESS_CUH

#define GPU_BLOCK_THREADS 1024
const int NUM_BOX_ELEMENT = 8; // left, top, right, bottom, confidence, class,

void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                  float* confidence_thresholds, float nms_threshold,
                                  float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                                  cudaStream_t stream);

#endif //POSTPROCESS_CUH
