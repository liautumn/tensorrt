#include "seg_postprocess.cuh"
#include "logger.h"

namespace seg {
    InstanceSegmentMap::InstanceSegmentMap(int width, int height) {
        this->width = width;
        this->height = height;
        checkRuntime(cudaMallocHost(&this->data, width * height));
    }

    InstanceSegmentMap::~InstanceSegmentMap() {
        if (this->data) {
            checkRuntime(cudaFreeHost(this->data));
            this->data = nullptr;
        }
        this->width = 0;
        this->height = 0;
    }

    __device__ __host__ void affine_project(float *matrix, float x, float y, float *ox, float *oy) {
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    static __global__ void decode_kernel_v8_Seg(float *predict, int num_bboxes, int num_classes,
                                            int output_cdim, float confidence_threshold,
                                            float *invert_affine_matrix, float *parray,
                                            int MAX_IMAGE_BOXES) {
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= num_bboxes) return;

        float *pitem = predict + output_cdim * position;
        float *class_confidence = pitem + 4;
        float confidence = *class_confidence++;
        int label = 0;
        for (int i = 1; i < num_classes; ++i, ++class_confidence) {
            if (*class_confidence > confidence) {
                confidence = *class_confidence;
                label = i;
            }
        }
        if (confidence < confidence_threshold) return;

        int index = atomicAdd(parray, 1);
        if (index >= MAX_IMAGE_BOXES) return;

        float cx = *pitem++;
        float cy = *pitem++;
        float width = *pitem++;
        float height = *pitem++;
        float left = cx - width * 0.5f;
        float top = cy - height * 0.5f;
        float right = cx + width * 0.5f;
        float bottom = cy + height * 0.5f;
        affine_project(invert_affine_matrix, left, top, &left, &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

        float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = label;
        *pout_item++ = 1; // 1 = keep, 0 = ignore
        *pout_item++ = position;
    }

    static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                    float btop, float bright, float bbottom) {
        float cleft = max(aleft, bleft);
        float ctop = max(atop, btop);
        float cright = min(aright, bright);
        float cbottom = min(abottom, bbottom);

        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if (c_area == 0.0f) return 0.0f;

        float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
        float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
        return c_area / (a_area + b_area - c_area);
    }

    static __global__ void fast_nms_kernel(float *bboxes, int MAX_IMAGE_BOXES, float threshold) {
        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min((int) *bboxes, MAX_IMAGE_BOXES);
        if (position >= count) return;

        // left, top, right, bottom, confidence, class, keepflag
        float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
        for (int i = 0; i < count; ++i) {
            float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
            if (i == position || pcurrent[5] != pitem[5]) continue;

            if (pitem[4] >= pcurrent[4]) {
                if (pitem[4] == pcurrent[4] && i < position) continue;

                float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                                    pitem[2], pitem[3]);

                if (iou > threshold) {
                    pcurrent[6] = 0; // 1=keep, 0=ignore
                    return;
                }
            }
        }
    }

    static __global__ void decode_single_mask_kernel(int left, int top, float *mask_weights,
                                                     float *mask_predict, int mask_width,
                                                     int mask_height, unsigned char *mask_out,
                                                     int mask_dim, int out_width, int out_height) {
        // mask_predict to mask_out
        // mask_weights @ mask_predict
        int dx = blockDim.x * blockIdx.x + threadIdx.x;
        int dy = blockDim.y * blockIdx.y + threadIdx.y;
        if (dx >= out_width || dy >= out_height) return;

        int sx = left + dx;
        int sy = top + dy;
        if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height) {
            mask_out[dy * out_width + dx] = 0;
            return;
        }

        float cumprod = 0;
        for (int ic = 0; ic < mask_dim; ++ic) {
            float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
            float wval = mask_weights[ic];
            cumprod += cval * wval;
        }

        float alpha = 1.0f / (1.0f + exp(-cumprod));
        mask_out[dy * out_width + dx] = alpha * 255;
    }

    void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                            int mask_width, int mask_height, unsigned char *mask_out,
                            int mask_dim, int out_width, int out_height, cudaStream_t stream) {
        // mask_weights is mask_dim(32 element) gpu pointer
        dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
        dim3 block(32, 32);

        checkKernel(decode_single_mask_kernel<<<grid, block, 0, stream>>>(
                        left, top, mask_weights, mask_predict, mask_width, mask_height, mask_out, mask_dim, out_width,
                        out_height));
    }

    static dim3 grid_dims(int numJobs) {
        int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
        return dim3((numJobs + numBlockThreads - 1) / static_cast<float>(numBlockThreads));
    }

    static dim3 block_dims(int numJobs) {
        return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    }

    void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                               float confidence_threshold, float nms_threshold,
                               float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                               cudaStream_t stream) {
        auto grid = grid_dims(num_bboxes);
        auto block = block_dims(num_bboxes);
        checkKernel(decode_kernel_v8_Seg<<<grid, block, 0, stream>>>(
                        predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
                        parray, MAX_IMAGE_BOXES));

        grid = grid_dims(MAX_IMAGE_BOXES);
        block = block_dims(MAX_IMAGE_BOXES);
        checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(parray, MAX_IMAGE_BOXES, nms_threshold));
    }
}
