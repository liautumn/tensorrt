#include "yolo26_c.h"

#include "c_api_error.h"
#include "export_common.h"
#include "logger.h"
#include "yolo26.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct yolo26_detector {
  explicit yolo26_detector(const char *engine_file, float confidence,
                           int gpu_device)
      : detector(engine_file, confidence, gpu_device) {}

  yolo26::Detector detector;
};

struct yolo26_result {
  std::vector<std::vector<yolo26_detection>> detections;
  yolo26_timing timing{};
};

static_assert(sizeof(yolo26_image) == 24, "unexpected yolo26_image ABI");
static_assert(sizeof(yolo26_detection) == 24,
              "unexpected yolo26_detection ABI");
static_assert(sizeof(yolo26_timing) == 16, "unexpected yolo26_timing ABI");

namespace {

thread_local char last_error[1024] = {};

} // namespace

namespace yolo26::detail {

void clear_c_api_error() noexcept { last_error[0] = '\0'; }

void set_c_api_error(const char *message) noexcept {
  std::snprintf(last_error, sizeof(last_error), "%s",
                message == nullptr ? "unknown error" : message);
  log(LogLevel::error, last_error);
}

} // namespace yolo26::detail

int32_t yolo26_detector_create(const char *engine_file_utf8, float confidence,
                               int32_t gpu_device,
                               yolo26_detector **out_detector) {
  if (out_detector != nullptr) {
    *out_detector = nullptr;
  }
  return yolo26::detail::c_api_call([&] {
    if (engine_file_utf8 == nullptr || engine_file_utf8[0] == '\0' ||
        out_detector == nullptr) {
      throw std::invalid_argument(
          "engine file and output detector are required");
    }
    *out_detector =
        new yolo26_detector(engine_file_utf8, confidence, gpu_device);
  });
}

void yolo26_detector_destroy(yolo26_detector *detector) { delete detector; }

int32_t yolo26_detector_predict(yolo26_detector *detector,
                                const yolo26_image *images, int32_t image_count,
                                yolo26_result **out_result) {
  if (out_result != nullptr) {
    *out_result = nullptr;
  }
  return yolo26::detail::c_api_call([&] {
    if (detector == nullptr || images == nullptr || image_count <= 0 ||
        out_result == nullptr) {
      throw std::invalid_argument("detector, images, positive image count and "
                                  "output result are required");
    }

    const std::vector<yolo26::Image> inputs =
        yolo26::export_api::make_images(images, image_count);

    std::vector<yolo26::Detections> native = detector->detector.predict(inputs);
    auto result = std::make_unique<yolo26_result>();
    result->detections.resize(native.size());
    for (std::size_t image_index = 0; image_index < native.size();
         ++image_index) {
      auto &target = result->detections[image_index];
      target.reserve(native[image_index].size());
      for (const yolo26::Detection &detection : native[image_index]) {
        target.push_back({detection.left, detection.top, detection.right,
                          detection.bottom, detection.confidence,
                          detection.class_id});
      }
    }
    const yolo26::Timing &timing = detector->detector.last_timing();
    result->timing = {timing.preprocess_ms, timing.inference_ms,
                      timing.postprocess_ms, timing.total_ms};
    *out_result = result.release();
  });
}

int32_t yolo26_result_get(const yolo26_result *result, int32_t image_index,
                          const yolo26_detection **out_detections,
                          int32_t *out_count) {
  if (out_detections != nullptr) {
    *out_detections = nullptr;
  }
  if (out_count != nullptr) {
    *out_count = 0;
  }
  return yolo26::detail::c_api_call([&] {
    if (result == nullptr || out_detections == nullptr ||
        out_count == nullptr || image_index < 0 ||
        static_cast<std::size_t>(image_index) >= result->detections.size()) {
      throw std::invalid_argument("invalid result image index");
    }
    const auto &detections =
        result->detections[static_cast<std::size_t>(image_index)];
    if (detections.size() >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error("too many detections for C API");
    }
    *out_detections = detections.empty() ? nullptr : detections.data();
    *out_count = static_cast<int32_t>(detections.size());
  });
}

int32_t yolo26_result_get_timing(const yolo26_result *result,
                                 yolo26_timing *out_timing) {
  if (out_timing != nullptr) {
    *out_timing = {};
  }
  return yolo26::detail::c_api_call([&] {
    if (result == nullptr || out_timing == nullptr) {
      throw std::invalid_argument("result and output timing are required");
    }
    *out_timing = result->timing;
  });
}

void yolo26_result_destroy(yolo26_result *result) { delete result; }

const char *yolo26_last_error(void) { return last_error; }
