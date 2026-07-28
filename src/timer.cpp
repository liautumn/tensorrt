#include "timer.h"

#include "cuda_utils.h"

namespace yolo26::detail {

CudaStageTimer::CudaStageTimer() {
  YOLO26_CHECK_CUDA(cudaEventCreate(&start_));
  try {
    YOLO26_CHECK_CUDA(cudaEventCreate(&stop_));
  } catch (...) {
    cudaEventDestroy(start_);
    throw;
  }
}

CudaStageTimer::~CudaStageTimer() {
  if (stop_ != nullptr) {
    cudaEventDestroy(stop_);
  }
  if (start_ != nullptr) {
    cudaEventDestroy(start_);
  }
}

void CudaStageTimer::start(cudaStream_t stream) {
  YOLO26_CHECK_CUDA(cudaEventRecord(start_, stream));
}

float CudaStageTimer::stop(cudaStream_t stream) {
  YOLO26_CHECK_CUDA(cudaEventRecord(stop_, stream));
  YOLO26_CHECK_CUDA(cudaEventSynchronize(stop_));
  float milliseconds = 0.0F;
  YOLO26_CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start_, stop_));
  return milliseconds;
}

PredictionTimer::PredictionTimer() : total_start_(Clock::now()) {}

void PredictionTimer::start_preprocess(cudaStream_t stream) {
  cuda_timer_.start(stream);
}

void PredictionTimer::stop_preprocess(cudaStream_t stream) {
  timing_.preprocess_ms = cuda_timer_.stop(stream);
}

void PredictionTimer::start_inference(cudaStream_t stream) {
  cuda_timer_.start(stream);
}

void PredictionTimer::stop_inference(cudaStream_t stream) {
  timing_.inference_ms = cuda_timer_.stop(stream);
}

void PredictionTimer::start_postprocess() { postprocess_start_ = Clock::now(); }

void PredictionTimer::stop_postprocess() {
  timing_.postprocess_ms = std::chrono::duration<float, std::milli>(
                               Clock::now() - postprocess_start_)
                               .count();
}

const Timing &PredictionTimer::finish() {
  timing_.total_ms =
      std::chrono::duration<float, std::milli>(Clock::now() - total_start_)
          .count();
  return timing_;
}

} // namespace yolo26::detail
