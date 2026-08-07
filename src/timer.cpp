// 引入 CUDA Event 计时器声明。
#include "timer.h"

// std::snprintf 用于格式化计时结果。
#include <cstdio>

// CUDA Event 的正常调用统一使用 checkRuntime，销毁失败只记录日志。
#include "validator.h"

namespace trt_timer
{

Timer::Timer()
{
    checkRuntime(cudaEventCreate(&start_));
    try
    {
        checkRuntime(cudaEventCreate(&stop_));
    }
    catch (...)
    {
        // 回滚不抛新的销毁异常，保留 stop event 创建失败这一原始错误。
        cudaEventDestroy(start_);
        start_ = nullptr;
        throw;
    }
}

Timer::~Timer()
{
    // 析构函数不能抛异常；销毁失败时只输出 CUDA 错误。
    if (stop_ != nullptr)
    {
        if (cudaError_t const status = cudaEventDestroy(stop_); status != cudaSuccess)
        {
            char message[256]{};
            std::snprintf(message, sizeof(message),
                "cudaEventDestroy(stop) failed: %s", cudaGetErrorString(status));
            Validator::log(message);
        }
    }
    if (start_ != nullptr)
    {
        if (cudaError_t const status = cudaEventDestroy(start_); status != cudaSuccess)
        {
            char message[256]{};
            std::snprintf(message, sizeof(message),
                "cudaEventDestroy(start) failed: %s", cudaGetErrorString(status));
            Validator::log(message);
        }
    }
}

void Timer::start(cudaStream_t stream)
{
    stream_ = stream;
    checkRuntime(cudaEventRecord(start_, stream_));
}

float Timer::stop(char const* prefix, bool print)
{
    checkRuntime(cudaEventRecord(stop_, stream_));
    checkRuntime(cudaEventSynchronize(stop_));

    float latency = 0.0F;
    checkRuntime(cudaEventElapsedTime(&latency, start_, stop_));

    if (print)
    {
        char message[256]{};
        std::snprintf(message, sizeof(message), "[%s]: %.3f ms", prefix, latency);
        Validator::info(message);
    }
    return latency;
}

} // namespace trt_timer
