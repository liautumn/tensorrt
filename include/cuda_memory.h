#ifndef YOLO26_CUDA_MEMORY_H
#define YOLO26_CUDA_MEMORY_H

#include "checked_math.h"
#include "cuda_utils.h"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace yolo26::detail {

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *resize(std::size_t count) {
    if (count <= capacity_) {
      return data_;
    }
    const std::size_t bytes =
        checked_multiply(count, sizeof(T), "CUDA buffer is too large");
    reset();
    YOLO26_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&data_), bytes));
    capacity_ = count;
    return data_;
  }

  T *data() noexcept { return data_; }

  void clear() noexcept { reset(); }

private:
  void reset() noexcept {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    data_ = nullptr;
    capacity_ = 0;
  }

  T *data_ = nullptr;
  std::size_t capacity_ = 0;
};

class CudaStream {
public:
  CudaStream() { YOLO26_CHECK_CUDA(cudaStreamCreate(&stream_)); }
  ~CudaStream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  cudaStream_t get() const noexcept { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
};

class CudaDeviceGuard {
public:
  explicit CudaDeviceGuard(int device) {
    YOLO26_CHECK_CUDA(cudaGetDevice(&previous_device_));
    if (previous_device_ != device) {
      YOLO26_CHECK_CUDA(cudaSetDevice(device));
      changed_ = true;
    }
  }

  ~CudaDeviceGuard() {
    if (changed_) {
      cudaSetDevice(previous_device_);
    }
  }

  CudaDeviceGuard(const CudaDeviceGuard &) = delete;
  CudaDeviceGuard &operator=(const CudaDeviceGuard &) = delete;

private:
  int previous_device_ = 0;
  bool changed_ = false;
};

class StreamCompletionGuard {
public:
  explicit StreamCompletionGuard(cudaStream_t stream) : stream_(stream) {}
  ~StreamCompletionGuard() {
    if (!finished_) {
      cudaStreamSynchronize(stream_);
    }
  }

  void finish() noexcept { finished_ = true; }

private:
  cudaStream_t stream_ = nullptr;
  bool finished_ = false;
};

} // namespace yolo26::detail

#endif // YOLO26_CUDA_MEMORY_H
