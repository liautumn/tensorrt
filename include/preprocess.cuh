#ifndef YOLO26_PREPROCESS_CUH
#define YOLO26_PREPROCESS_CUH

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace yolo26::detail {

struct LetterboxTransform {
  double scale = 1.0;
  int resized_width = 0;
  int resized_height = 0;
  int left = 0;
  int top = 0;

  static LetterboxTransform compute(int source_width, int source_height,
                                    int target_width, int target_height);
  void to_source(float &x, float &y) const;
};

void launch_preprocess(const std::uint8_t *source, std::size_t source_stride,
                       int source_width, int source_height, float *destination,
                       int target_width, int target_height,
                       LetterboxTransform transform, cudaStream_t stream);

} // namespace yolo26::detail

#endif // YOLO26_PREPROCESS_CUH
