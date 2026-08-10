// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "cuda_raii.h"

#include <string_view>

namespace trt_timer
{

// 使用同一条 CUDA stream 上的两个事件测量 GPU 操作耗时。
class Timer
{
public:
    Timer();
    ~Timer() = default;

    Timer(Timer const&) = delete;
    Timer& operator=(Timer const&) = delete;
    Timer(Timer&&) noexcept = default;
    Timer& operator=(Timer&&) noexcept = default;

    // 在 stream 上记录起始事件。
    void start(cudaStream_t stream = nullptr);
    // 记录并等待结束事件，返回两个事件之间的毫秒数。
    [[nodiscard]] float stop(std::string_view prefix = "Timer", bool print = true);

private:
    CudaEvent start_{};
    CudaEvent stop_{};
    cudaStream_t stream_{};
};

} // namespace trt_timer
