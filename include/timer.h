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
    // 构造一对 CUDA event；事件创建失败时通过统一检查抛出异常。
    Timer();
    // 事件由 unique_ptr 自动销毁，析构本身不需要额外同步调用。
    ~Timer() = default;

    // Event 句柄不能复制，否则会出现重复销毁；移动可转移唯一所有权。
    Timer(Timer const&) = delete;
    Timer& operator=(Timer const&) = delete;
    Timer(Timer&&) noexcept = default;
    Timer& operator=(Timer&&) noexcept = default;

    // 在指定 stream 上记录起始事件，并保存该 stream 供 stop() 使用。
    void start(cudaStream_t stream = nullptr);
    // 调用前必须先 start()；函数在保存的 stream 上记录并等待结束事件，返回两个事件之间的 GPU 毫秒数。
    // prefix/print 只影响可选日志，不影响返回值。
    [[nodiscard]] float stop(std::string_view prefix = "Timer", bool print = true);

private:
    // 本次测量的起始时间点；事件由当前 Timer 独占。
    CudaEvent start_{};
    // 本次测量的结束时间点；stop() 会等待它完成。
    CudaEvent stop_{};
    // start() 最近一次绑定的 stream，stop() 必须使用同一条队列。
    cudaStream_t stream_{};
};

} // namespace trt_timer
