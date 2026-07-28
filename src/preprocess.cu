#include "preprocess.cuh"

#include "cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace yolo26::detail {
namespace {

constexpr float kPaddingValue = 114.0F;

__device__ float3 read_bgr(const std::uint8_t *source, std::size_t stride,
                           int width, int height, int x, int y) {
  x = x < 0 ? 0 : (x >= width ? width - 1 : x);
  y = y < 0 ? 0 : (y >= height ? height - 1 : y);
  const std::uint8_t *pixel = source + static_cast<std::size_t>(y) * stride +
                              static_cast<std::size_t>(x) * 3;
  return make_float3(pixel[0], pixel[1], pixel[2]);
}

__device__ float interpolate(float a, float b, float c, float d, float weight_x,
                             float weight_y) {
  const float upper = a + (b - a) * weight_x;
  const float lower = c + (d - c) * weight_x;
  return floorf(upper + (lower - upper) * weight_y + 0.5F);
}

__global__ void preprocess_kernel(const std::uint8_t *source,
                                  std::size_t source_stride, int source_width,
                                  int source_height, float *destination,
                                  int target_width, int target_height,
                                  LetterboxTransform transform) {
  const int target_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int target_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (target_x >= target_width || target_y >= target_height) {
    return;
  }

  const std::size_t area =
      static_cast<std::size_t>(target_width) * target_height;
  const std::size_t index =
      static_cast<std::size_t>(target_y) * target_width + target_x;
  const int resized_x = target_x - transform.left;
  const int resized_y = target_y - transform.top;
  if (resized_x < 0 || resized_x >= transform.resized_width || resized_y < 0 ||
      resized_y >= transform.resized_height) {
    const float padding = kPaddingValue / 255.0F;
    destination[index] = padding;
    destination[area + index] = padding;
    destination[2 * area + index] = padding;
    return;
  }

  // This is OpenCV INTER_LINEAR's half-pixel resize mapping.
  const float source_x =
      (resized_x + 0.5F) * source_width / transform.resized_width - 0.5F;
  const float source_y =
      (resized_y + 0.5F) * source_height / transform.resized_height - 0.5F;
  const int x0 = static_cast<int>(floorf(source_x));
  const int y0 = static_cast<int>(floorf(source_y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float weight_x = source_x - x0;
  const float weight_y = source_y - y0;

  const float3 p00 =
      read_bgr(source, source_stride, source_width, source_height, x0, y0);
  const float3 p01 =
      read_bgr(source, source_stride, source_width, source_height, x1, y0);
  const float3 p10 =
      read_bgr(source, source_stride, source_width, source_height, x0, y1);
  const float3 p11 =
      read_bgr(source, source_stride, source_width, source_height, x1, y1);

  const float blue =
      interpolate(p00.x, p01.x, p10.x, p11.x, weight_x, weight_y);
  const float green =
      interpolate(p00.y, p01.y, p10.y, p11.y, weight_x, weight_y);
  const float red = interpolate(p00.z, p01.z, p10.z, p11.z, weight_x, weight_y);

  destination[index] = red / 255.0F;
  destination[area + index] = green / 255.0F;
  destination[2 * area + index] = blue / 255.0F;
}

} // namespace

LetterboxTransform LetterboxTransform::compute(int source_width,
                                               int source_height,
                                               int target_width,
                                               int target_height) {
  const double scale =
      std::min(target_width / static_cast<double>(source_width),
               target_height / static_cast<double>(source_height));
  const int resized_width =
      static_cast<int>(std::nearbyint(source_width * scale));
  const int resized_height =
      static_cast<int>(std::nearbyint(source_height * scale));
  if (resized_width <= 0 || resized_height <= 0) {
    throw std::invalid_argument(
        "image aspect ratio is too extreme for the model input size");
  }
  const double horizontal_padding = (target_width - resized_width) / 2.0;
  const double vertical_padding = (target_height - resized_height) / 2.0;
  return {
      scale,
      resized_width,
      resized_height,
      static_cast<int>(std::round(horizontal_padding - 0.1)),
      static_cast<int>(std::round(vertical_padding - 0.1)),
  };
}

void LetterboxTransform::to_source(float &x, float &y) const {
  x = static_cast<float>((x - left) / scale);
  y = static_cast<float>((y - top) / scale);
}

void launch_preprocess(const std::uint8_t *source, std::size_t source_stride,
                       int source_width, int source_height, float *destination,
                       int target_width, int target_height,
                       LetterboxTransform transform, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((target_width + block.x - 1) / block.x,
                  (target_height + block.y - 1) / block.y);
  preprocess_kernel<<<grid, block, 0, stream>>>(
      source, source_stride, source_width, source_height, destination,
      target_width, target_height, transform);
  YOLO26_CHECK_CUDA(cudaGetLastError());
}

} // namespace yolo26::detail
