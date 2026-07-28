#ifndef YOLO26_POSTPROCESS_H
#define YOLO26_POSTPROCESS_H

#include "preprocess.cuh"
#include "yolo26.h"

#include <vector>

namespace yolo26::detail {

std::vector<Detections>
decode_detections(const float *output, int detections_per_image,
                  float confidence_threshold, const std::vector<Image> &images,
                  const std::vector<LetterboxTransform> &transforms);

} // namespace yolo26::detail

#endif // YOLO26_POSTPROCESS_H
