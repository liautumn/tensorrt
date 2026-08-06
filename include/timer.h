#pragma once

// CUDA Runtime 接口：提供 cudaEvent_t、cudaStream_t 和事件计时函数。
#include <cuda_runtime_api.h>

namespace trt_timer
{

// 使用同一条 CUDA stream 上的两个事件测量 GPU 操作耗时。
class Timer
{
public:
    Timer();
    ~Timer();

    Timer(Timer const&) = delete;
    Timer& operator=(Timer const&) = delete;

    // 在 stream 上记录起始事件。
    void start(cudaStream_t stream = nullptr);
    // 记录并等待结束事件，返回两个事件之间的毫秒数。
    float stop(char const* prefix = "Timer", bool print = true);

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
    cudaStream_t stream_{};
};

} // namespace trt_timer
