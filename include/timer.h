#ifndef YOLO26_TIMER_H
#define YOLO26_TIMER_H

#include "yolo26.h"

#include <cuda_runtime_api.h>

#include <chrono>

namespace yolo26::detail {

class CudaStageTimer {
public:
  CudaStageTimer();
  ~CudaStageTimer();

  CudaStageTimer(const CudaStageTimer &) = delete;
  CudaStageTimer &operator=(const CudaStageTimer &) = delete;

  void start(cudaStream_t stream);
  float stop(cudaStream_t stream);

private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

class PredictionTimer {
public:
  PredictionTimer();

  void start_preprocess(cudaStream_t stream);
  void stop_preprocess(cudaStream_t stream);
  void start_inference(cudaStream_t stream);
  void stop_inference(cudaStream_t stream);
  void start_postprocess();
  void stop_postprocess();
  const Timing &finish();

private:
  using Clock = std::chrono::steady_clock;

  Clock::time_point total_start_;
  Clock::time_point postprocess_start_;
  CudaStageTimer cuda_timer_;
  Timing timing_;
};

} // namespace yolo26::detail

#endif // YOLO26_TIMER_H
