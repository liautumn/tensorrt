#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace yolo26::detail {

std::vector<Detections>
decode_detections(const float *output, int detections_per_image,
                  float confidence_threshold, const std::vector<Image> &images,
                  const std::vector<LetterboxTransform> &transforms) {
  std::vector<Detections> result(images.size());
  for (std::size_t batch_index = 0; batch_index < images.size();
       ++batch_index) {
    Detections &detections = result[batch_index];
    detections.reserve(detections_per_image);
    for (int row_index = 0; row_index < detections_per_image; ++row_index) {
      const std::size_t offset =
          (batch_index * static_cast<std::size_t>(detections_per_image) +
           static_cast<std::size_t>(row_index)) *
          6;
      const float *row = output + offset;
      const float rounded_class = std::round(row[5]);
      if (!std::isfinite(row[4]) || row[4] <= confidence_threshold ||
          !std::isfinite(row[5]) || row[5] < 0.0F ||
          row[5] > static_cast<float>(std::numeric_limits<int>::max()) ||
          std::fabs(row[5] - rounded_class) > 1e-3F) {
        continue;
      }

      float left = row[0];
      float top = row[1];
      float right = row[2];
      float bottom = row[3];
      if (!std::isfinite(left) || !std::isfinite(top) ||
          !std::isfinite(right) || !std::isfinite(bottom)) {
        continue;
      }
      transforms[batch_index].to_source(left, top);
      transforms[batch_index].to_source(right, bottom);

      const Image &image = images[batch_index];
      left = std::clamp(left, 0.0F, static_cast<float>(image.width));
      top = std::clamp(top, 0.0F, static_cast<float>(image.height));
      right = std::clamp(right, 0.0F, static_cast<float>(image.width));
      bottom = std::clamp(bottom, 0.0F, static_cast<float>(image.height));
      if (right <= left || bottom <= top) {
        continue;
      }

      detections.push_back(
          {left, top, right, bottom, row[4], static_cast<int>(rounded_class)});
    }
  }
  return result;
}

} // namespace yolo26::detail
