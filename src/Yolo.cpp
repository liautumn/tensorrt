#include <cuda_runtime_api.h>
#include <iostream>
#include "Infer.h"
#include "Yolo.h"
#include <cstring>
#include "Memory.h"
#include "Preprocess.cuh"
#include "Postprocess.cuh"
#include "Logger.h"

namespace yolo
{
    using namespace std;

    // keepflag, row_index(output)
    const int MAX_IMAGE_BOXES = 1024;

    inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }

    struct AffineMatrix
    {
        float i2d[6]; // image to dst(network), 2x3 matrix
        float d2i[6]; // dst to image, 2x3 matrix

        void compute(const tuple<int, int>& from, const tuple<int, int>& to)
        {
            float scale_x = get<0>(to) / static_cast<float>(get<0>(from));
            float scale_y = get<1>(to) / static_cast<float>(get<1>(from));
            float scale = min(scale_x, scale_y);
            i2d[0] = scale;
            i2d[1] = 0;
            i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;
            i2d[4] = scale;
            i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

            double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
            D = D != 0. ? double(1.) / D : double(0.);
            double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
            double b1 = -A11 * i2d[2] - A12 * i2d[5];
            double b2 = -A21 * i2d[2] - A22 * i2d[5];

            d2i[0] = A11;
            d2i[1] = A12;
            d2i[2] = b1;
            d2i[3] = A21;
            d2i[4] = A22;
            d2i[5] = b2;
        }
    };

    class InferImpl : public Infer
    {
    public:
        shared_ptr<trt::Infer> trt_;
        string engine_file_;
        float confidence_threshold_;
        void* cuda_stream_;
        float nms_threshold_;
        vector<shared_ptr<trt_memory::Memory<unsigned char>>> preprocess_buffers_;
        trt_memory::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
        int network_input_width_, network_input_height_;
        Norm normalize_;
        vector<int> bbox_head_dims_;
        int num_classes_ = 0;
        bool isDynamic_model_ = false;

        virtual ~InferImpl()
        {
            cudaStreamDestroy(static_cast<cudaStream_t>(cuda_stream_));
        };

        void adjust_memory(int batch_size)
        {
            // the inference batch_size
            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            input_buffer_.gpu(batch_size * input_numel);
            bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
            output_boxarray_.gpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
            output_boxarray_.cpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

            if (static_cast<int>(preprocess_buffers_.size()) < batch_size)
            {
                for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                    preprocess_buffers_.push_back(make_shared<trt_memory::Memory<unsigned char>>());
            }
        }

        void preprocess(int ibatch,
                        const Image& image,
                        shared_ptr<trt_memory::Memory<unsigned char>> preprocess_buffer,
                        AffineMatrix& affine,
                        void* stream = nullptr)
        {
            affine.compute(make_tuple(image.width, image.height),
                           make_tuple(network_input_width_, network_input_height_));

            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            float* input_device = input_buffer_.gpu() + ibatch * input_numel;
            size_t size_image = image.width * image.height * 3;
            size_t size_matrix = upbound(sizeof(affine.d2i), 32);

            uint8_t* gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
            auto* affine_matrix_device = reinterpret_cast<float*>(gpu_workspace);
            uint8_t* image_device = gpu_workspace + size_matrix;

            uint8_t* cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
            auto* affine_matrix_host = reinterpret_cast<float*>(cpu_workspace);
            uint8_t* image_host = cpu_workspace + size_matrix;

            // speed up
            auto stream_ = static_cast<cudaStream_t>(stream);
            memcpy(image_host, image.bgrptr, size_image);
            memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
            checkRuntime(
                cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                cudaMemcpyHostToDevice, stream_));

            warp_affine_bilinear_and_normalize_plane(image_device, image.width * 3, image.width,
                                                     image.height, input_device, network_input_width_,
                                                     network_input_height_, affine_matrix_device, 114,
                                                     normalize_, stream_);
        }

        bool load(const string& engine_file,
                  float confidence_threshold,
                  float nms_threshold, void* stream = nullptr)
        {
            trt_ = trt::load(engine_file);
            if (trt_ == nullptr) return false;

            trt_->print();

            this->cuda_stream_ = stream;
            this->confidence_threshold_ = confidence_threshold;
            this->nms_threshold_ = nms_threshold;

            auto inputName = trt_->name(0);
            auto outputName = trt_->name(1);
            auto input_dim = trt_->static_dims(inputName);
            bbox_head_dims_ = trt_->static_dims(outputName);
            network_input_width_ = input_dim[3];
            network_input_height_ = input_dim[2];
            isDynamic_model_ = trt_->has_dynamic_dim();

            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4;
            return true;
        }

        BoxArray forward(const Image& image,
                         void* stream = nullptr) override
        {
            auto output = forwards({image}, stream);
            if (output.empty()) return {};
            return output[0];
        }

        vector<BoxArray> forwards(const vector<Image>& images,
                                  void* stream = nullptr) override
        {
            int num_image = images.size();
            if (num_image == 0) return {};
            auto inputName = trt_->name(0);
            auto input_dims = trt_->static_dims(inputName);
            int infer_batch_size = input_dims[0];
            if (infer_batch_size != num_image)
            {
                if (isDynamic_model_)
                {
                    infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!trt_->set_run_dims(inputName, input_dims)) return {};
                }
                else
                {
                    if (infer_batch_size < num_image)
                    {
                        INFO(
                            "When using static shape model, number of images[%d] must be "
                            "less than or equal to the maximum batch[%d].",
                            num_image, infer_batch_size);
                        return {};
                    }
                }
            }
            adjust_memory(infer_batch_size);

            vector<AffineMatrix> affine_matrixs(num_image);
            auto stream_ = static_cast<cudaStream_t>(stream);
            for (int i = 0; i < num_image; ++i)
                preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

            float* bbox_output_device = bbox_predict_.gpu();

            vector<void*> bindings{input_buffer_.gpu(), bbox_output_device};

            if (!trt_->forward(bindings, stream))
            {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            for (int ib = 0; ib < num_image; ++ib)
            {
                float* boxarray_device =
                    output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
                float* affine_matrix_device = reinterpret_cast<float*>(preprocess_buffers_[ib]->gpu());
                float* image_based_bbox_output =
                    bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
                checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
                decode_kernel_invoker(image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                                      bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                                      affine_matrix_device, boxarray_device, MAX_IMAGE_BOXES, stream_);
            }
            checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(),
                output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
            checkRuntime(cudaStreamSynchronize(stream_));

            vector<BoxArray> arrout(num_image);
            for (int ib = 0; ib < num_image; ++ib)
            {
                float* parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
                int count = min(MAX_IMAGE_BOXES, static_cast<int>(*parray));
                BoxArray& output = arrout[ib];
                output.reserve(count);
                for (int i = 0; i < count; ++i)
                {
                    float* pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                    int label = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1)
                    {
                        Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                        output.emplace_back(result_object_box);
                    }
                }
            }
            return arrout;
        }
    };

    shared_ptr<Infer> load(const string& engine_file,
                           const float confidence_threshold,
                           const float nms_threshold,
                           void* stream)
    {
        auto* impl = new InferImpl();
        if (!impl->load(engine_file, confidence_threshold, nms_threshold, stream))
        {
            delete impl;
            impl = nullptr;
        }
        return shared_ptr<InferImpl>(impl);
    }
}; // namespace yolo
