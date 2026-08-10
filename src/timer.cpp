// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

// 引入 CUDA Event 计时器声明。
#include "timer.h"

#include <algorithm>
// std::snprintf 用于格式化计时结果。
#include <cstdio>

// CUDA Event 的正常调用统一使用 checkRuntime，销毁失败只记录日志。
#include "validator.h"

namespace trt_timer
{

Timer::Timer()
    : start_(createCudaEvent())
    , stop_(createCudaEvent())
{
}

void Timer::start(cudaStream_t stream)
{
    stream_ = stream;
    checkRuntime(cudaEventRecord(start_.get(), stream_));
}

float Timer::stop(std::string_view const prefix, bool const print)
{
    checkRuntime(cudaEventRecord(stop_.get(), stream_));
    checkRuntime(cudaEventSynchronize(stop_.get()));

    float latency = 0.0F;
    checkRuntime(cudaEventElapsedTime(&latency, start_.get(), stop_.get()));

    if (print)
    {
        constexpr std::size_t maxPrefixLength = 200;
        int const prefixLength = static_cast<int>(std::min(prefix.size(), maxPrefixLength));
        char const* const prefixData = prefix.empty() ? "" : prefix.data();

        char message[256]{};
        std::snprintf(message, sizeof(message), "[%.*s]: %.3f ms",
            prefixLength, prefixData, latency);
        Validator::info(message);
    }
    return latency;
}

} // namespace trt_timer
