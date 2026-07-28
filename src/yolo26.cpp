#include "yolo26.h"

#include "checked_math.h"
#include "cuda_memory.h"
#include "cuda_utils.h"
#include "logger.h"
#include "postprocess.h"
#include "preprocess.cuh"
#include "tensorrt_engine.h"
#include "timer.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace yolo26 {
namespace {

std::size_t checked_image_bytes(const Image &image) {
  if (image.bgr == nullptr || image.width <= 0 || image.height <= 0) {
    throw std::invalid_argument("image must contain non-empty BGR8 data");
  }
  const std::size_t packed_stride = detail::checked_multiply(
      static_cast<std::size_t>(image.width), 3, "image row is too large");
  if (image.row_stride() < packed_stride) {
    throw std::invalid_argument("image stride is smaller than width * 3");
  }
  return detail::checked_multiply(packed_stride,
                                  static_cast<std::size_t>(image.height),
                                  "image is too large");
}

bool has_dynamic_dimension(const std::vector<int> &shape) {
  return std::any_of(shape.begin(), shape.end(),
                     [](int dimension) { return dimension == -1; });
}

std::string timing_message(std::size_t batch, const Timing &timing) {
  std::ostringstream message;
  message << std::fixed << std::setprecision(3) << "batch=" << batch
          << ", preprocess=" << timing.preprocess_ms
          << " ms, inference=" << timing.inference_ms
          << " ms, postprocess=" << timing.postprocess_ms
          << " ms, total=" << timing.total_ms << " ms";
  return message.str();
}

} // namespace

class Detector::Impl {
public:
  Impl(const std::string &engine_file, float confidence_threshold,
       int gpu_device)
      : gpu_device_(gpu_device), confidence_threshold_(confidence_threshold) {
    if (!std::isfinite(confidence_threshold) || confidence_threshold < 0.0F ||
        confidence_threshold > 1.0F) {
      throw std::invalid_argument("confidence threshold must be in [0, 1]");
    }

    detail::CudaDeviceGuard device_guard(gpu_device_);
    try {
      engine_ = std::make_unique<detail::TensorRTEngine>(engine_file);
      inspect_engine();
      stream_ = std::make_unique<detail::CudaStream>();
    } catch (...) {
      stream_.reset();
      engine_.reset();
      throw;
    }
    detail::log(detail::LogLevel::info,
                "loaded YOLO26 detection engine: " + engine_file);
  }

  ~Impl() {
    try {
      detail::CudaDeviceGuard device_guard(gpu_device_);
      source_buffer_.clear();
      input_buffer_.clear();
      output_buffer_.clear();
      stream_.reset();
      engine_.reset();
    } catch (...) {
      // 析构函数不能向调用方抛出 CUDA 异常。
    }
  }

  std::vector<Detections> predict(const std::vector<Image> &images) {
    if (images.empty()) {
      last_timing_ = {};
      return {};
    }
    if (images.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("batch is too large");
    }
    const int batch = static_cast<int>(images.size());
    validate_batch(batch);

    detail::CudaDeviceGuard device_guard(gpu_device_);
    detail::StreamCompletionGuard completion_guard(stream_->get());
    detail::PredictionTimer timer;

    std::size_t source_capacity = 0;
    for (const Image &image : images) {
      source_capacity = std::max(source_capacity, checked_image_bytes(image));
    }
    source_buffer_.resize(source_capacity);

    const std::vector<int> input_shape{batch, 3, input_height_, input_width_};
    if (dynamic_input_) {
      engine_->set_input_shape(input_shape);
    }

    const std::size_t input_area = detail::checked_multiply(
        static_cast<std::size_t>(input_height_),
        static_cast<std::size_t>(input_width_), "model input is too large");
    const std::size_t image_elements =
        detail::checked_multiply(input_area, 3, "model input is too large");
    const std::size_t input_elements =
        detail::checked_multiply(static_cast<std::size_t>(batch),
                                 image_elements, "batch input is too large");
    input_buffer_.resize(input_elements);

    std::vector<detail::LetterboxTransform> transforms;
    transforms.reserve(images.size());
    timer.start_preprocess(stream_->get());
    for (std::size_t index = 0; index < images.size(); ++index) {
      const Image &image = images[index];
      const std::size_t packed_stride =
          static_cast<std::size_t>(image.width) * 3;
      YOLO26_CHECK_CUDA(cudaMemcpy2DAsync(
          source_buffer_.data(), packed_stride, image.bgr, image.row_stride(),
          packed_stride, image.height, cudaMemcpyHostToDevice, stream_->get()));

      const auto transform = detail::LetterboxTransform::compute(
          image.width, image.height, input_width_, input_height_);
      transforms.push_back(transform);
      detail::launch_preprocess(
          source_buffer_.data(), packed_stride, image.width, image.height,
          input_buffer_.data() + index * image_elements, input_width_,
          input_height_, transform, stream_->get());
    }
    timer.stop_preprocess(stream_->get());

    const std::vector<int> output_shape =
        engine_->runtime_shape(engine_->output_name());
    validate_output_shape(output_shape, batch);
    const std::size_t output_rows = detail::checked_multiply(
        static_cast<std::size_t>(batch),
        static_cast<std::size_t>(output_shape[1]), "batch output is too large");
    const std::size_t output_elements = detail::checked_multiply(
        output_rows, static_cast<std::size_t>(output_shape[2]),
        "batch output is too large");
    output_buffer_.resize(output_elements);
    host_output_.resize(output_elements);

    timer.start_inference(stream_->get());
    engine_->set_tensor_addresses(input_buffer_.data(), output_buffer_.data());
    engine_->enqueue(stream_->get());
    YOLO26_CHECK_CUDA(
        cudaMemcpyAsync(host_output_.data(), output_buffer_.data(),
                        detail::checked_multiply(output_elements, sizeof(float),
                                                 "batch output is too large"),
                        cudaMemcpyDeviceToHost, stream_->get()));
    timer.stop_inference(stream_->get());
    completion_guard.finish();

    timer.start_postprocess();
    std::vector<Detections> result =
        detail::decode_detections(host_output_.data(), output_shape[1],
                                  confidence_threshold_, images, transforms);
    timer.stop_postprocess();
    last_timing_ = timer.finish();
    detail::log(detail::LogLevel::info,
                timing_message(images.size(), last_timing_));
    return result;
  }

  int input_width() const noexcept { return input_width_; }
  int input_height() const noexcept { return input_height_; }
  int min_batch_size() const noexcept { return min_batch_size_; }
  int max_batch_size() const noexcept { return max_batch_size_; }
  const Timing &last_timing() const noexcept { return last_timing_; }

private:
  void validate_batch(int batch) const {
    if (dynamic_batch_) {
      if (batch < min_batch_size_ || batch > max_batch_size_) {
        throw std::invalid_argument("batch size must be in [" +
                                    std::to_string(min_batch_size_) + ", " +
                                    std::to_string(max_batch_size_) + "]");
      }
    } else if (batch != max_batch_size_) {
      throw std::invalid_argument("fixed-batch engine requires exactly " +
                                  std::to_string(max_batch_size_) + " images");
    }
  }

  void inspect_engine() {
    if (engine_->data_type(engine_->input_name()) !=
            nvinfer1::DataType::kFLOAT ||
        engine_->data_type(engine_->output_name()) !=
            nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error(
          "YOLO26 engine input and output tensors must use float32 I/O");
    }
    if (engine_->tensor_format(engine_->input_name()) !=
            nvinfer1::TensorFormat::kLINEAR ||
        engine_->tensor_format(engine_->output_name()) !=
            nvinfer1::TensorFormat::kLINEAR) {
      throw std::runtime_error(
          "YOLO26 engine input and output tensors must use linear I/O layout");
    }

    const std::vector<int> input =
        engine_->declared_shape(engine_->input_name());
    const std::vector<int> output =
        engine_->declared_shape(engine_->output_name());
    if (input.size() != 4 || input[1] != 3) {
      throw std::runtime_error("expected YOLO26 input [N,3,H,W], got " +
                               detail::format_shape(input));
    }
    if (output.size() != 3 || (output[2] != 6 && output[2] != -1)) {
      throw std::runtime_error(
          "expected end-to-end YOLO26 output [N,K,6], got " +
          detail::format_shape(output));
    }

    dynamic_input_ = has_dynamic_dimension(input);
    dynamic_batch_ = input[0] == -1;
    if (dynamic_input_) {
      const std::vector<int> minimum = engine_->profile_shape(
          engine_->input_name(), nvinfer1::OptProfileSelector::kMIN);
      const std::vector<int> optimum = engine_->profile_shape(
          engine_->input_name(), nvinfer1::OptProfileSelector::kOPT);
      const std::vector<int> maximum = engine_->profile_shape(
          engine_->input_name(), nvinfer1::OptProfileSelector::kMAX);
      if (minimum.size() != 4 || optimum.size() != 4 || maximum.size() != 4) {
        throw std::runtime_error(
            "invalid TensorRT optimization profile for YOLO26 input");
      }
      min_batch_size_ = input[0] == -1 ? minimum[0] : input[0];
      max_batch_size_ = input[0] == -1 ? maximum[0] : input[0];
      input_height_ = input[2] == -1 ? optimum[2] : input[2];
      input_width_ = input[3] == -1 ? optimum[3] : input[3];
    } else {
      min_batch_size_ = max_batch_size_ = input[0];
      input_height_ = input[2];
      input_width_ = input[3];
    }

    if (min_batch_size_ <= 0 || max_batch_size_ < min_batch_size_ ||
        input_height_ <= 0 || input_width_ <= 0) {
      throw std::runtime_error(
          "invalid YOLO26 input shape or optimization profile");
    }
    detail::checked_multiply(static_cast<std::size_t>(input_height_),
                             static_cast<std::size_t>(input_width_),
                             "YOLO26 input dimensions are too large");
  }

  static void validate_output_shape(const std::vector<int> &shape, int batch) {
    if (shape.size() != 3 || shape[0] != batch || shape[1] <= 0 ||
        shape[2] != 6) {
      throw std::runtime_error("expected runtime output [N,K,6], got " +
                               detail::format_shape(shape));
    }
  }

  int gpu_device_ = 0;
  float confidence_threshold_ = 0.25F;
  int min_batch_size_ = 1;
  int max_batch_size_ = 1;
  int input_width_ = 0;
  int input_height_ = 0;
  bool dynamic_input_ = false;
  bool dynamic_batch_ = false;
  Timing last_timing_;
  std::unique_ptr<detail::TensorRTEngine> engine_;
  std::unique_ptr<detail::CudaStream> stream_;
  detail::DeviceBuffer<std::uint8_t> source_buffer_;
  detail::DeviceBuffer<float> input_buffer_;
  detail::DeviceBuffer<float> output_buffer_;
  std::vector<float> host_output_;
};

Detector::Detector(const std::string &engine_file, float confidence_threshold,
                   int gpu_device)
    : impl_(std::make_unique<Impl>(engine_file, confidence_threshold,
                                   gpu_device)) {}

Detector::~Detector() = default;
Detector::Detector(Detector &&) noexcept = default;
Detector &Detector::operator=(Detector &&) noexcept = default;

Detections Detector::predict(const Image &image) {
  std::vector<Detections> result = impl_->predict({image});
  return result.empty() ? Detections{} : std::move(result.front());
}

std::vector<Detections> Detector::predict(const std::vector<Image> &images) {
  return impl_->predict(images);
}

int Detector::input_width() const noexcept { return impl_->input_width(); }
int Detector::input_height() const noexcept { return impl_->input_height(); }
int Detector::min_batch_size() const noexcept {
  return impl_->min_batch_size();
}
int Detector::max_batch_size() const noexcept {
  return impl_->max_batch_size();
}
const Timing &Detector::last_timing() const noexcept {
  return impl_->last_timing();
}

} // namespace yolo26
