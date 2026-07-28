#ifndef YOLO26_EXPORT_COMMON_H
#define YOLO26_EXPORT_COMMON_H

#include "cpm.h"
#include "yolo26.h"
#include "yolo26_c.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace yolo26::export_api {

using AsyncDetector = cpm::Instance<Detections, Image, Detector>;

inline std::vector<Image> make_images(const yolo26_image *images,
                                      int32_t image_count) {
  if (images == nullptr || image_count <= 0) {
    throw std::invalid_argument("images and positive image count are required");
  }
  std::vector<Image> result;
  result.reserve(static_cast<std::size_t>(image_count));
  for (int32_t index = 0; index < image_count; ++index) {
    if (images[index].stride >
        static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
      throw std::invalid_argument("image stride is too large");
    }
    result.emplace_back(images[index].bgr, images[index].width,
                        images[index].height,
                        static_cast<std::size_t>(images[index].stride));
  }
  return result;
}
} // namespace yolo26::export_api

#endif // YOLO26_EXPORT_COMMON_H
