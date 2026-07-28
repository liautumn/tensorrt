#ifndef YOLO26_CUDA_UTILS_H
#define YOLO26_CUDA_UTILS_H

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace yolo26::detail {

inline void check_cuda(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(operation) +
                           " failed: " + cudaGetErrorName(status) + " (" +
                           cudaGetErrorString(status) + ")");
}

} // namespace yolo26::detail

#define YOLO26_CHECK_CUDA(operation)                                           \
  ::yolo26::detail::check_cuda((operation), #operation)

#endif // YOLO26_CUDA_UTILS_H
