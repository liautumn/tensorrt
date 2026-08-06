// 引入 CUDA Event 计时器声明。
#include "timer.h"

// std::fprintf 和 stderr 用于输出计时结果及析构阶段的 CUDA 错误。
#include <cstdio>
// std::runtime_error 用于报告 CUDA Event 创建、记录或同步失败。
#include <stdexcept>
// std::string 用于拼接 CUDA 错误信息。
#include <string>

namespace
{

void checkCuda(cudaError_t status, char const* operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

} // namespace

namespace trt_timer
{

Timer::Timer()
{
    checkCuda(cudaEventCreate(&start_), "cudaEventCreate(start)");
    try
    {
        checkCuda(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
    }
    catch (...)
    {
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
            std::fprintf(stderr, "cudaEventDestroy(stop) failed: %s\n", cudaGetErrorString(status));
        }
    }
    if (start_ != nullptr)
    {
        if (cudaError_t const status = cudaEventDestroy(start_); status != cudaSuccess)
        {
            std::fprintf(stderr, "cudaEventDestroy(start) failed: %s\n", cudaGetErrorString(status));
        }
    }
}

void Timer::start(cudaStream_t stream)
{
    stream_ = stream;
    checkCuda(cudaEventRecord(start_, stream_), "cudaEventRecord(start)");
}

float Timer::stop(char const* prefix, bool print)
{
    checkCuda(cudaEventRecord(stop_, stream_), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stop_), "cudaEventSynchronize(stop)");

    float latency = 0.0F;
    checkCuda(cudaEventElapsedTime(&latency, start_, stop_), "cudaEventElapsedTime");

    if (print)
    {
        std::printf("[%s]: %.3f ms\n", prefix, latency);
    }
    return latency;
}

} // namespace trt_timer
