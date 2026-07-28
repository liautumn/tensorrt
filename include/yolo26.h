#ifndef YOLO26_H
#define YOLO26_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace yolo26 {

struct Image {
  const std::uint8_t *bgr = nullptr;
  int width = 0;
  int height = 0;
  std::size_t stride = 0;

  Image() = default;
  Image(const std::uint8_t *data, int image_width, int image_height,
        std::size_t row_stride = 0)
      : bgr(data), width(image_width), height(image_height),
        stride(row_stride) {}

  std::size_t row_stride() const {
    return stride == 0 ? static_cast<std::size_t>(width) * 3 : stride;
  }
};

struct Detection {
  float left = 0.0F;
  float top = 0.0F;
  float right = 0.0F;
  float bottom = 0.0F;
  float confidence = 0.0F;
  int class_id = -1;
};

using Detections = std::vector<Detection>;

struct Timing {
  float preprocess_ms = 0.0F;
  float inference_ms = 0.0F;
  float postprocess_ms = 0.0F;
  float total_ms = 0.0F;
};

class Detector {
public:
  Detector(const std::string &engine_file, float confidence_threshold = 0.25F,
           int gpu_device = 0);
  ~Detector();

  Detector(const Detector &) = delete;
  Detector &operator=(const Detector &) = delete;
  Detector(Detector &&) noexcept;
  Detector &operator=(Detector &&) noexcept;

  Detections predict(const Image &image);
  std::vector<Detections> predict(const std::vector<Image> &images);

  int input_width() const noexcept;
  int input_height() const noexcept;
  int min_batch_size() const noexcept;
  int max_batch_size() const noexcept;
  const Timing &last_timing() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace yolo26

#endif // YOLO26_H
