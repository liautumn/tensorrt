#include <cuda_runtime_api.h>
#include "yolo.h"
#include <algorithm>
#include "infer.h"
#include <iostream>
#include "cls_postprocess.h"
#include "memory.h"
#include "logger.h"
#include "preprocess.cuh"
#include "detect_postprocess.cuh"

namespace yolo {
    using namespace std;

    // keepflag, row_index(output)
    const int MAX_IMAGE_BOXES = 1024;

    inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }

    class InferImpl : public Infer {
    public:
        shared_ptr<trt::infer> trt_;
        string engine_file_;
        float confidence_threshold_;
        void *cuda_stream_;
        float nms_threshold_;
        vector<shared_ptr<trt_memory::Memory<unsigned char> > > preprocess_buffers_;
        trt_memory::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
        int network_input_width_, network_input_height_;
        Norm normalize_;
        vector<int> bbox_head_dims_;
        bool isDynamic_model_ = false;

        vector<int> segment_head_dims_;
        trt_memory::Memory<float> segment_predict_;
        vector<shared_ptr<trt_memory::Memory<unsigned char> > > box_segment_cache_;

        virtual ~InferImpl() {
            cudaStreamDestroy(static_cast<cudaStream_t>(cuda_stream_));
        };

        void preprocess(int ibatch,
                        const Image &image,
                        shared_ptr<trt_memory::Memory<unsigned char> > preprocess_buffer,
                        AffineMatrix &affine,
                        void *stream = nullptr) {
            affine.compute(make_tuple(image.width, image.height),
                           make_tuple(network_input_width_, network_input_height_));

            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            float *input_device = input_buffer_.gpu() + ibatch * input_numel;
            size_t size_image = image.width * image.height * 3;
            size_t size_matrix = upbound(sizeof(affine.d2i), 32);

            uint8_t *gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
            auto *affine_matrix_device = reinterpret_cast<float *>(gpu_workspace);
            uint8_t *image_device = gpu_workspace + size_matrix;

            uint8_t *cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
            auto *affine_matrix_host = reinterpret_cast<float *>(cpu_workspace);
            uint8_t *image_host = cpu_workspace + size_matrix;

            // speed up
            auto stream_ = static_cast<cudaStream_t>(stream);
            memcpy(image_host, image.bgrptr, size_image);
            memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
            checkRuntime(
                cudaMemcpyAsync(image_device,
                    image_host,
                    size_image,
                    cudaMemcpyHostToDevice,
                    stream_)
            );
            checkRuntime(
                cudaMemcpyAsync(affine_matrix_device,
                    affine_matrix_host,
                    sizeof(affine.d2i),
                    cudaMemcpyHostToDevice,
                    stream_)
            );

            warp_affine_bilinear_and_normalize_plane(image_device,
                                                     image.width * 3,
                                                     image.width,
                                                     image.height, input_device,
                                                     network_input_width_,
                                                     network_input_height_,
                                                     affine_matrix_device,
                                                     114,
                                                     normalize_,
                                                     stream_);
        }

        bool load(int gpu_device,
                  const string &engine_file,
                  float confidence_threshold,
                  float nms_threshold,
                  void *stream = nullptr) {
            checkRuntime(cudaSetDevice(gpu_device));
            trt_ = trt::load(engine_file);
            if (trt_ == nullptr) return false;

            trt_->print();

            this->cuda_stream_ = stream;
            this->confidence_threshold_ = confidence_threshold;
            this->nms_threshold_ = nms_threshold;

            auto input_dim = trt_->static_dims(trt_->name(0));
            bbox_head_dims_ = trt_->static_dims(trt_->name(1));
            network_input_width_ = input_dim[3];
            network_input_height_ = input_dim[2];
            isDynamic_model_ = trt_->has_dynamic_dim();
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);

            // segment_head_dims_ = trt_->static_dims(trt_->name(2));
            return true;
        }

        detect::BoxArray detect_forward(const Image &image,
                                        void *stream = nullptr) override {
            auto output = detect_forwards({image}, stream);
            if (output.empty()) return {};
            return output[0];
        }

        vector<detect::BoxArray> detect_forwards(const vector<Image> &images,
                                                 void *stream = nullptr) override {
            auto num_classes_ = bbox_head_dims_[2] - 4;
            int num_image = images.size();
            if (num_image == 0) return {};
            auto inputName = trt_->name(0);
            auto input_dims = trt_->static_dims(inputName);
            int infer_batch_size = input_dims[0];
            if (infer_batch_size != num_image) {
                if (isDynamic_model_) {
                    infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!trt_->set_run_dims(inputName, input_dims)) return {};
                } else {
                    if (infer_batch_size < num_image) {
                        INFO(
                            "When using static shape model, number of images[%d] must be "
                            "less than or equal to the maximum batch[%d].",
                            num_image, infer_batch_size);
                        return {};
                    }
                }
            }

            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            input_buffer_.gpu(infer_batch_size * input_numel);
            bbox_predict_.gpu(infer_batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
            output_boxarray_.gpu(infer_batch_size * (32 + MAX_IMAGE_BOXES * detect::NUM_BOX_ELEMENT));
            output_boxarray_.cpu(infer_batch_size * (32 + MAX_IMAGE_BOXES * detect::NUM_BOX_ELEMENT));

            if (static_cast<int>(preprocess_buffers_.size()) < infer_batch_size) {
                for (int i = preprocess_buffers_.size(); i < infer_batch_size; ++i)
                    preprocess_buffers_.push_back(make_shared<trt_memory::Memory<unsigned char> >());
            }

            vector<AffineMatrix> affine_matrixs(num_image);
            auto stream_ = static_cast<cudaStream_t>(stream);
            for (int i = 0; i < num_image; ++i)
                preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

            float *bbox_output_device = bbox_predict_.gpu();

            vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};

            if (!trt_->forward(bindings, 2, stream)) {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            for (int ib = 0; ib < num_image; ++ib) {
                float *boxarray_device =
                        output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * detect::NUM_BOX_ELEMENT);
                float *affine_matrix_device = reinterpret_cast<float *>(preprocess_buffers_[ib]->gpu());
                float *image_based_bbox_output = bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
                checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
                detect::decode_kernel_invoker(image_based_bbox_output,
                                              bbox_head_dims_[1],
                                              num_classes_,
                                              bbox_head_dims_[2],
                                              confidence_threshold_,
                                              nms_threshold_,
                                              affine_matrix_device,
                                              boxarray_device,
                                              MAX_IMAGE_BOXES,
                                              stream_);
            }
            checkRuntime(
                cudaMemcpyAsync(
                    output_boxarray_.cpu(),
                    output_boxarray_.gpu(),
                    output_boxarray_.gpu_bytes(),
                    cudaMemcpyDeviceToHost,
                    stream_)
            );
            checkRuntime(cudaStreamSynchronize(stream_));

            vector<detect::BoxArray> arrout(num_image);
            for (int ib = 0; ib < num_image; ++ib) {
                float *parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * detect::NUM_BOX_ELEMENT);
                int count = min(MAX_IMAGE_BOXES, static_cast<int>(*parray));
                detect::BoxArray &output = arrout[ib];
                output.reserve(count);
                for (int i = 0; i < count; ++i) {
                    float *pbox = parray + 1 + i * detect::NUM_BOX_ELEMENT;
                    int label = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1) {
                        detect::Box result_object_box(pbox[0],
                                                      pbox[1],
                                                      pbox[2],
                                                      pbox[3],
                                                      pbox[4],
                                                      label);
                        output.emplace_back(result_object_box);
                    }
                }
            }
            return arrout;
        }

        seg::BoxArray seg_forward(const Image &image,
                                  void *stream = nullptr) override {
            auto output = seg_forwards({image}, stream);
            if (output.empty()) return {};
            return output[0];
        }

        vector<seg::BoxArray> seg_forwards(const vector<Image> &images,
                                           void *stream = nullptr) override {
            auto num_classes_ = bbox_head_dims_[2] - 4 - segment_head_dims_[1];

            int num_image = images.size();
            if (num_image == 0) return {};
            auto inputName = trt_->name(0);
            auto input_dims = trt_->static_dims(inputName);
            int infer_batch_size = input_dims[0];
            if (infer_batch_size != num_image) {
                if (isDynamic_model_) {
                    infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!trt_->set_run_dims(inputName, input_dims)) return {};
                } else {
                    if (infer_batch_size < num_image) {
                        INFO(
                            "When using static shape model, number of images[%d] must be "
                            "less than or equal to the maximum batch[%d].",
                            num_image, infer_batch_size);
                        return {};
                    }
                }
            }

            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            input_buffer_.gpu(infer_batch_size * input_numel);
            bbox_predict_.gpu(infer_batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
            output_boxarray_.gpu(infer_batch_size * (32 + MAX_IMAGE_BOXES * seg::NUM_BOX_ELEMENT));
            output_boxarray_.cpu(infer_batch_size * (32 + MAX_IMAGE_BOXES * seg::NUM_BOX_ELEMENT));
            segment_predict_.gpu(infer_batch_size * segment_head_dims_[1] * segment_head_dims_[2] *
                                 segment_head_dims_[3]);
            if (static_cast<int>(preprocess_buffers_.size()) < num_image) {
                for (int i = preprocess_buffers_.size(); i < num_image; ++i)
                    preprocess_buffers_.push_back(make_shared<trt_memory::Memory<unsigned char> >());
            }

            vector<AffineMatrix> affine_matrixs(num_image);
            auto stream_ = static_cast<cudaStream_t>(stream);
            for (int i = 0; i < num_image; ++i)
                preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

            float *bbox_output_device = bbox_predict_.gpu();

            vector<void *> bindings{input_buffer_.gpu(), bbox_output_device, segment_predict_.gpu()};;

            if (!trt_->forward(bindings, 3, stream)) {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            for (int ib = 0; ib < num_image; ++ib) {
                float *boxarray_device =
                        output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * seg::NUM_BOX_ELEMENT);
                float *affine_matrix_device = reinterpret_cast<float *>(preprocess_buffers_[ib]->gpu());
                float *image_based_bbox_output = bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
                checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
                seg::decode_kernel_invoker(image_based_bbox_output,
                                           bbox_head_dims_[1],
                                           num_classes_,
                                           bbox_head_dims_[2],
                                           confidence_threshold_,
                                           nms_threshold_,
                                           affine_matrix_device,
                                           boxarray_device,
                                           MAX_IMAGE_BOXES,
                                           stream_);
            }
            checkRuntime(
                cudaMemcpyAsync(
                    output_boxarray_.cpu(),
                    output_boxarray_.gpu(),
                    output_boxarray_.gpu_bytes(),
                    cudaMemcpyDeviceToHost,
                    stream_)
            );
            checkRuntime(cudaStreamSynchronize(stream_));

            vector<seg::BoxArray> arrout(num_image);
            int imemory = 0;
            for (int ib = 0; ib < num_image; ++ib) {
                float *parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * seg::NUM_BOX_ELEMENT);
                int count = min(MAX_IMAGE_BOXES, (int) *parray);
                seg::BoxArray &output = arrout[ib];
                output.reserve(count);
                for (int i = 0; i < count; ++i) {
                    float *pbox = parray + 1 + i * seg::NUM_BOX_ELEMENT;
                    int label = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1) {
                        seg::Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                        int row_index = pbox[7];
                        int mask_dim = segment_head_dims_[1];
                        float *mask_weights = bbox_output_device +
                                              (ib * bbox_head_dims_[1] + row_index) * bbox_head_dims_[2] +
                                              num_classes_ + 4;

                        float *mask_head_predict = segment_predict_.gpu();
                        float left, top, right, bottom;
                        float *i2d = affine_matrixs[ib].i2d;
                        seg::affine_project(i2d, pbox[0], pbox[1], &left, &top);
                        seg::affine_project(i2d, pbox[2], pbox[3], &right, &bottom);

                        float box_width = right - left;
                        float box_height = bottom - top;

                        float scale_to_predict_x = segment_head_dims_[3] / static_cast<float>(network_input_width_);
                        float scale_to_predict_y = segment_head_dims_[2] / static_cast<float>(network_input_height_);

                        int mask_out_width = box_width * scale_to_predict_x + 0.5f;
                        int mask_out_height = box_height * scale_to_predict_y + 0.5f;

                        if (mask_out_width > 0 && mask_out_height > 0) {
                            if (imemory >= (int) box_segment_cache_.size()) {
                                box_segment_cache_.push_back(std::make_shared<trt_memory::Memory<unsigned char> >());
                            }

                            int bytes_of_mask_out = mask_out_width * mask_out_height;
                            auto box_segment_output_memory = box_segment_cache_[imemory];
                            result_object_box.seg =
                                    make_shared<seg::InstanceSegmentMap>(mask_out_width, mask_out_height);

                            unsigned char *mask_out_device = box_segment_output_memory->gpu(bytes_of_mask_out);
                            unsigned char *mask_out_host = result_object_box.seg->data;
                            seg::decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y, mask_weights,
                                                    mask_head_predict + ib * segment_head_dims_[1] *
                                                    segment_head_dims_[2] *
                                                    segment_head_dims_[3],
                                                    segment_head_dims_[3], segment_head_dims_[2], mask_out_device,
                                                    mask_dim, mask_out_width, mask_out_height, stream_);
                            checkRuntime(cudaMemcpyAsync(mask_out_host, mask_out_device,
                                box_segment_output_memory->gpu_bytes(),
                                cudaMemcpyDeviceToHost, stream_));
                        }
                        result_object_box.seg->left = left * scale_to_predict_x;
                        result_object_box.seg->top = top * scale_to_predict_y;
                        output.emplace_back(result_object_box);
                    }
                }
            }
            checkRuntime(cudaStreamSynchronize(stream_));
            return arrout;
        }

        obb::BoxArray obb_forward(const Image &image,
                                  void *stream = nullptr) override {
            auto output = obb_forwards({image}, stream);
            if (output.empty()) return {};
            return output[0];
        }

        vector<obb::BoxArray> obb_forwards(const vector<Image> &images,
                                           void *stream = nullptr) override {
            auto num_classes_ = bbox_head_dims_[2] - 5;
            int num_image = images.size();
            if (num_image == 0) return {};
            auto inputName = trt_->name(0);
            auto input_dims = trt_->static_dims(inputName);
            int infer_batch_size = input_dims[0];
            if (infer_batch_size != num_image) {
                if (isDynamic_model_) {
                    infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!trt_->set_run_dims(inputName, input_dims)) return {};
                } else {
                    if (infer_batch_size < num_image) {
                        INFO(
                            "When using static shape model, number of images[%d] must be "
                            "less than or equal to the maximum batch[%d].",
                            num_image, infer_batch_size);
                        return {};
                    }
                }
            }

            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            input_buffer_.gpu(infer_batch_size * input_numel);
            bbox_predict_.gpu(infer_batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
            output_boxarray_.gpu(infer_batch_size * (32 + MAX_IMAGE_BOXES * obb::NUM_BOX_ELEMENT));
            output_boxarray_.cpu(infer_batch_size * (32 + MAX_IMAGE_BOXES * obb::NUM_BOX_ELEMENT));
            if (static_cast<int>(preprocess_buffers_.size()) < infer_batch_size) {
                for (int i = preprocess_buffers_.size(); i < infer_batch_size; ++i)
                    preprocess_buffers_.push_back(make_shared<trt_memory::Memory<unsigned char> >());
            }

            vector<AffineMatrix> affine_matrixs(num_image);
            auto stream_ = static_cast<cudaStream_t>(stream);
            for (int i = 0; i < num_image; ++i)
                preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

            float *bbox_output_device = bbox_predict_.gpu();

            vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};

            if (!trt_->forward(bindings, 2, stream)) {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            for (int ib = 0; ib < num_image; ++ib) {
                float *boxarray_device =
                        output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * obb::NUM_BOX_ELEMENT);
                float *affine_matrix_device = reinterpret_cast<float *>(preprocess_buffers_[ib]->gpu());
                float *image_based_bbox_output = bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
                checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
                obb::decode_kernel_invoker(image_based_bbox_output,
                                           bbox_head_dims_[1],
                                           num_classes_,
                                           bbox_head_dims_[2],
                                           confidence_threshold_,
                                           nms_threshold_,
                                           affine_matrix_device,
                                           boxarray_device,
                                           MAX_IMAGE_BOXES,
                                           stream_);
            }
            checkRuntime(
                cudaMemcpyAsync(
                    output_boxarray_.cpu(),
                    output_boxarray_.gpu(),
                    output_boxarray_.gpu_bytes(),
                    cudaMemcpyDeviceToHost,
                    stream_)
            );
            checkRuntime(cudaStreamSynchronize(stream_));

            vector<obb::BoxArray> arrout(num_image);
            for (int ib = 0; ib < num_image; ++ib) {
                float *parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * obb::NUM_BOX_ELEMENT);
                int count = min(MAX_IMAGE_BOXES, static_cast<int>(*parray));
                obb::BoxArray &output = arrout[ib];
                output.reserve(count);
                for (int i = 0; i < count; ++i) {
                    float *pbox = parray + 1 + i * obb::NUM_BOX_ELEMENT;
                    // cx, cy, w, h, angle, confidence, class_label, keepflag
                    int keepflag = pbox[7];
                    if (keepflag == 1) {
                        output.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], pbox[5], pbox[6]);
                    }
                }
            }
            return arrout;
        }

        cls::ProbArray cls_forward(const Image &image,
                                   void *stream = nullptr) override {
            auto output = cls_forwards({image}, stream);
            if (output.empty()) return {};
            return output[0];
        }

        vector<cls::ProbArray> cls_forwards(const vector<Image> &images,
                                            void *stream = nullptr) override {
            int num_image = images.size();
            if (num_image == 0) return {};
            auto inputName = trt_->name(0);
            auto input_dims = trt_->static_dims(inputName);
            int infer_batch_size = input_dims[0];
            if (infer_batch_size != num_image) {
                if (isDynamic_model_) {
                    infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!trt_->set_run_dims(inputName, input_dims)) return {};
                } else {
                    if (infer_batch_size < num_image) {
                        INFO(
                            "When using static shape model, number of images[%d] must be "
                            "less than or equal to the maximum batch[%d].",
                            num_image, infer_batch_size);
                        return {};
                    }
                }
            }

            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            input_buffer_.gpu(infer_batch_size * input_numel);
            bbox_predict_.gpu(infer_batch_size * bbox_head_dims_[1]);
            bbox_predict_.cpu(infer_batch_size * bbox_head_dims_[1]);
            if (static_cast<int>(preprocess_buffers_.size()) < infer_batch_size) {
                for (int i = preprocess_buffers_.size(); i < infer_batch_size; ++i)
                    preprocess_buffers_.push_back(make_shared<trt_memory::Memory<unsigned char> >());
            }

            vector<AffineMatrix> affine_matrixs(num_image);
            auto stream_ = static_cast<cudaStream_t>(stream);
            for (int i = 0; i < num_image; ++i)
                preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

            vector<void *> bindings{input_buffer_.gpu(), bbox_predict_.gpu()};

            if (!trt_->forward(bindings, 2, stream)) {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            checkRuntime(
                cudaMemcpyAsync(
                    bbox_predict_.cpu(),
                    bbox_predict_.gpu(),
                    bbox_predict_.gpu_bytes(),
                    cudaMemcpyDeviceToHost,
                    stream_)
            );
            checkRuntime(cudaStreamSynchronize(stream_));

            vector<cls::ProbArray> arrout(num_image);
            for (int ib = 0; ib < num_image; ++ib) {
                cls::ProbArray &prob_array = arrout[ib];
                float *parray = bbox_predict_.cpu() + ib * bbox_head_dims_[1];
                for (int i = 0; i < bbox_head_dims_[1]; ++i) {
                    if (parray[i] > confidence_threshold_) {
                        prob_array.emplace_back(i, parray[i]);
                    }
                }
                // int label = max_element(parray, parray + bbox_head_dims_[1]) - parray;
                // float confidence = parray[label];
            }

            return arrout;
        }

        pose::BoxArray pose_forward(const Image &image,
                                    void *stream = nullptr) override {
            auto output = pose_forwards({image}, stream);
            if (output.empty()) return {};
            return output[0];
        }

        vector<pose::BoxArray> pose_forwards(const vector<Image> &images,
                                             void *stream = nullptr) override {
            int num_image = images.size();
            if (num_image == 0) return {};
            auto inputName = trt_->name(0);
            auto input_dims = trt_->static_dims(inputName);
            int infer_batch_size = input_dims[0];
            if (infer_batch_size != num_image) {
                if (isDynamic_model_) {
                    infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!trt_->set_run_dims(inputName, input_dims)) return {};
                } else {
                    if (infer_batch_size < num_image) {
                        INFO(
                            "When using static shape model, number of images[%d] must be "
                            "less than or equal to the maximum batch[%d].",
                            num_image, infer_batch_size);
                        return {};
                    }
                }
            }

            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            input_buffer_.gpu(infer_batch_size * input_numel);
            bbox_predict_.gpu(infer_batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
            output_boxarray_.gpu(infer_batch_size * (32 + MAX_IMAGE_BOXES * pose::NUM_BOX_ELEMENT));
            output_boxarray_.cpu(infer_batch_size * (32 + MAX_IMAGE_BOXES * pose::NUM_BOX_ELEMENT));

            if (static_cast<int>(preprocess_buffers_.size()) < infer_batch_size) {
                for (int i = preprocess_buffers_.size(); i < infer_batch_size; ++i)
                    preprocess_buffers_.push_back(make_shared<trt_memory::Memory<unsigned char> >());
            }

            vector<AffineMatrix> affine_matrixs(num_image);
            auto stream_ = static_cast<cudaStream_t>(stream);
            for (int i = 0; i < num_image; ++i)
                preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

            float *bbox_output_device = bbox_predict_.gpu();

            vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};

            if (!trt_->forward(bindings, 2, stream)) {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            for (int ib = 0; ib < num_image; ++ib) {
                float *boxarray_device =
                        output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * pose::NUM_BOX_ELEMENT);
                float *affine_matrix_device = reinterpret_cast<float *>(preprocess_buffers_[ib]->gpu());
                float *image_based_bbox_output = bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
                checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
                pose::decode_kernel_invoker(image_based_bbox_output,
                                            bbox_head_dims_[1],
                                            bbox_head_dims_[2],
                                            confidence_threshold_,
                                            nms_threshold_,
                                            affine_matrix_device,
                                            boxarray_device,
                                            MAX_IMAGE_BOXES,
                                            stream_);
            }
            checkRuntime(
                cudaMemcpyAsync(
                    output_boxarray_.cpu(),
                    output_boxarray_.gpu(),
                    output_boxarray_.gpu_bytes(),
                    cudaMemcpyDeviceToHost,
                    stream_)
            );
            checkRuntime(cudaStreamSynchronize(stream_));

            vector<pose::BoxArray> arrout(num_image);
            for (int ib = 0; ib < num_image; ++ib) {
                float *parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * pose::NUM_BOX_ELEMENT);
                int count = min(MAX_IMAGE_BOXES, static_cast<int>(*parray));
                pose::BoxArray &output = arrout[ib];
                output.reserve(count);
                for (int i = 0; i < count; ++i) {
                    float *pbox = parray + 1 + i * pose::NUM_BOX_ELEMENT;
                    int keepflag = pbox[5];
                    if (keepflag == 1) {
                        pose::Box box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4]);
                        float *pkeypoint_start = pbox + 6;
                        float *pkeypoint_end = pkeypoint_start + 3 * pose::NUM_KEYPOINTS;
                        box.keypoints.insert(box.keypoints.end(), reinterpret_cast<cv::Point3f *>(pkeypoint_start),
                                             reinterpret_cast<cv::Point3f *>(pkeypoint_end));
                        output.emplace_back(box);
                    }
                }
            }
            return arrout;
        }
    };

    shared_ptr<Infer> load(const string &engine_file,
                           const float confidence_threshold,
                           const float nms_threshold,
                           const int gpu_device,
                           void *stream) {
        auto impl = std::make_shared<InferImpl>();
        if (!impl->load(gpu_device, engine_file, confidence_threshold, nms_threshold, stream)) {
            return nullptr;
        }
        return impl;
    }
}

; // namespace yolo
