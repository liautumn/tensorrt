#include "infer.h"
#include "yolo.h"
#include <cuda_runtime_api.h>
#include "preprocess.cuh"
#include "postprocess.cuh"

namespace yolo {
    using namespace std;

    // keepflag, row_index(output)
    const int MAX_IMAGE_BOXES = 1024;

    inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }

    struct AffineMatrix {
        float i2d[6]; // image to dst(network), 2x3 matrix
        float d2i[6]; // dst to image, 2x3 matrix

        void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) {
            float scale_x = get<0>(to) / (float) get<0>(from);
            float scale_y = get<1>(to) / (float) get<1>(from);
            float scale = std::min(scale_x, scale_y);
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

    class InferImpl : public Infer {
    public:
        shared_ptr<trt::Infer> trt_;
        string engine_file_;
        float confidence_threshold_;
        float nms_threshold_;
        vector<shared_ptr<trt::Memory<unsigned char> > > preprocess_buffers_;
        trt::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
        int network_input_width_, network_input_height_;
        Norm normalize_;
        vector<int> bbox_head_dims_;
        int num_classes_ = 0;
        bool isdynamic_model_ = false;

        virtual ~InferImpl() = default;

        void adjust_memory(int batch_size) {
            // the inference batch_size
            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            input_buffer_.gpu(batch_size * input_numel);
            bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
            output_boxarray_.gpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
            output_boxarray_.cpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

            if ((int) preprocess_buffers_.size() < batch_size) {
                for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                    preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char> >());
            }
        }

        void preprocess(int ibatch, const Image &image,
                        shared_ptr<trt::Memory<unsigned char> > preprocess_buffer, AffineMatrix &affine,
                        void *stream = nullptr) {
            affine.compute(make_tuple(image.width, image.height),
                           make_tuple(network_input_width_, network_input_height_));

            size_t input_numel = network_input_width_ * network_input_height_ * 3;
            float *input_device = input_buffer_.gpu() + ibatch * input_numel;
            size_t size_image = image.width * image.height * 3;
            size_t size_matrix = upbound(sizeof(affine.d2i), 32);
            uint8_t *gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
            float *affine_matrix_device = (float *) gpu_workspace;
            uint8_t *image_device = gpu_workspace + size_matrix;

            uint8_t *cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
            float *affine_matrix_host = (float *) cpu_workspace;
            uint8_t *image_host = cpu_workspace + size_matrix;

            // speed up
            cudaStream_t stream_ = (cudaStream_t) stream;
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

        bool load(const string &engine_file, float confidence_threshold, float nms_threshold) {
            trt_ = trt::load(engine_file);
            if (trt_ == nullptr) return false;

            trt_->print();

            this->confidence_threshold_ = confidence_threshold;
            this->nms_threshold_ = nms_threshold;

            auto input_dim = trt_->static_dims(0);
            bbox_head_dims_ = trt_->static_dims(1);
            network_input_width_ = input_dim[3];
            network_input_height_ = input_dim[2];
            isdynamic_model_ = trt_->has_dynamic_dim();

            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4;
            return true;
        }

        virtual BoxArray forward(const Image &image, void *stream = nullptr) override {
            auto output = forwards({image}, stream);
            if (output.empty()) return {};
            return output[0];
        }

        virtual vector<BoxArray> forwards(const vector<Image> &images, void *stream = nullptr) override {
            int num_image = images.size();
            if (num_image == 0) return {};

            auto input_dims = trt_->static_dims(0);
            int infer_batch_size = input_dims[0];
            if (infer_batch_size != num_image) {
                if (isdynamic_model_) {
                    infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!trt_->set_run_dims(0, input_dims)) return {};
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
            adjust_memory(infer_batch_size);

            vector<AffineMatrix> affine_matrixs(num_image);
            cudaStream_t stream_ = (cudaStream_t) stream;
            for (int i = 0; i < num_image; ++i)
                preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

            float *bbox_output_device = bbox_predict_.gpu();
            vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};

            if (!trt_->forward(bindings, stream)) {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            for (int ib = 0; ib < num_image; ++ib) {
                float *boxarray_device =
                        output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
                float *affine_matrix_device = (float *) preprocess_buffers_[ib]->gpu();
                float *image_based_bbox_output =
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
            for (int ib = 0; ib < num_image; ++ib) {
                float *parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
                int count = min(MAX_IMAGE_BOXES, (int) *parray);
                BoxArray &output = arrout[ib];
                output.reserve(count);
                for (int i = 0; i < count; ++i) {
                    float *pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                    int label = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1) {
                        Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                        output.emplace_back(result_object_box);
                    }
                }
            }
            return arrout;
        }
    };

    Infer *loadraw(const std::string &engine_file, float confidence_threshold,
                   float nms_threshold) {
        InferImpl *impl = new InferImpl();
        if (!impl->load(engine_file, confidence_threshold, nms_threshold)) {
            delete impl;
            impl = nullptr;
        }
        return impl;
    }

    shared_ptr<Infer> load(const string &engine_file, float confidence_threshold,
                           float nms_threshold) {
        return std::shared_ptr<InferImpl>(
            (InferImpl *) loadraw(engine_file, confidence_threshold, nms_threshold));
    }

    std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
        const int h_i = static_cast<int>(h * 6);
        const float f = h * 6 - h_i;
        const float p = v * (1 - s);
        const float q = v * (1 - f * s);
        const float t = v * (1 - (1 - f) * s);
        float r, g, b;
        switch (h_i) {
            case 0:
                r = v, g = t, b = p;
                break;
            case 1:
                r = q, g = v, b = p;
                break;
            case 2:
                r = p, g = v, b = t;
                break;
            case 3:
                r = p, g = q, b = v;
                break;
            case 4:
                r = t, g = p, b = v;
                break;
            case 5:
                r = v, g = p, b = q;
                break;
            default:
                r = 1, g = 1, b = 1;
                break;
        }
        return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
                          static_cast<uint8_t>(r * 255));
    }

    std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
        float h_plane = ((((unsigned int) id << 2) ^ 0x937151) % 100) / 100.0f;
        float s_plane = ((((unsigned int) id << 3) ^ 0x315793) % 100) / 100.0f;
        return hsv2bgr(h_plane, s_plane, 1);
    }
}; // namespace yolo
