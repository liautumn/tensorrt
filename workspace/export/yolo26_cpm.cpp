#include "export_common.h"

#include "c_api_error.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct yolo26_cpm {
  yolo26::export_api::AsyncDetector instance;
};

int32_t yolo26_cpm_create(const char *engine_file_utf8, float confidence,
                          int32_t gpu_device, int32_t max_batch_size,
                          yolo26_cpm **out_cpm) {
  if (out_cpm != nullptr) {
    *out_cpm = nullptr;
  }
  return yolo26::detail::c_api_call([&] {
    if (engine_file_utf8 == nullptr || engine_file_utf8[0] == '\0' ||
        max_batch_size <= 0 || out_cpm == nullptr) {
      throw std::invalid_argument(
          "engine file, positive max batch and output CPM are required");
    }
    const std::string engine_file(engine_file_utf8);
    const int requested_batch_size = max_batch_size;
    auto cpm = std::make_unique<yolo26_cpm>();
    cpm->instance.start(
        [engine_file, confidence, gpu_device, requested_batch_size] {
          auto detector = std::make_shared<yolo26::Detector>(
              engine_file, confidence, gpu_device);
          if (detector->min_batch_size() != 1) {
            throw std::invalid_argument(
                "CPM requires an engine whose minimum batch size is 1");
          }
          if (requested_batch_size > detector->max_batch_size()) {
            throw std::invalid_argument(
                "CPM max batch size exceeds the engine profile maximum");
          }
          return detector;
        },
        static_cast<std::size_t>(max_batch_size));
    *out_cpm = cpm.release();
  });
}

int32_t yolo26_cpm_predict_one(yolo26_cpm *cpm, const yolo26_image *image,
                               const yolo26_detection **detections,
                               int32_t *count) {
  if (detections != nullptr) {
    *detections = nullptr;
  }
  if (count != nullptr) {
    *count = 0;
  }
  return yolo26::detail::c_api_call([&] {
    if (cpm == nullptr || image == nullptr || detections == nullptr ||
        count == nullptr) {
      throw std::invalid_argument(
          "CPM, image, output detections and count are required");
    }
    const std::vector<yolo26::Image> inputs =
        yolo26::export_api::make_images(image, 1);
    const yolo26::Detections native =
        cpm->instance.commit(inputs.front()).get();
    auto &thread_results = []() -> std::vector<yolo26_detection> & {
      thread_local std::vector<yolo26_detection> value;
      return value;
    }();
    if (native.size() >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error("too many detections for C API");
    }
    thread_results.clear();
    thread_results.reserve(native.size());
    for (const auto &item : native) {
      thread_results.push_back({item.left, item.top, item.right, item.bottom,
                                item.confidence, item.class_id});
    }
    *detections = thread_results.empty() ? nullptr : thread_results.data();
    *count = static_cast<int32_t>(thread_results.size());
  });
}

void yolo26_cpm_destroy(yolo26_cpm *cpm) { delete cpm; }
