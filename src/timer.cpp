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
    // 先创建起始事件，再创建结束事件；成员初始化顺序与声明顺序一致。
    : start_(createCudaEvent())
    , stop_(createCudaEvent())
{
    // 两个事件在构造线程的当前 CUDA device 上创建，后续应绑定同一设备的 Model stream。
}

void Timer::start(cudaStream_t stream)
{
    // 保存本次测量的 stream；stop() 必须在同一条 stream 上记录结束事件。
    stream_ = stream;
    // 把起始事件排入目标 stream，事件之前的 GPU 工作不会计入本次测量。
    checkRuntime(cudaEventRecord(start_.get(), stream_));
}

float Timer::stop(std::string_view const prefix, bool const print)
{
    // 把结束事件排在同一条 stream 的末尾，覆盖 start 到此处的 GPU 工作。
    checkRuntime(cudaEventRecord(stop_.get(), stream_));
    // 等待结束事件完成，确保后面的 elapsedTime 读取到有效时间戳。
    checkRuntime(cudaEventSynchronize(stop_.get()));

    // CUDA EventElapsedTime 返回毫秒；先初始化结果，便于错误路径保持确定值。
    float latency = 0.0F;
    // 计算两个事件时间戳之间的 GPU elapsed time，而不是 CPU wall-clock 时间。
    checkRuntime(cudaEventElapsedTime(&latency, start_.get(), stop_.get()));

    // print=false 用于调用方只取数值、不重复输出日志的场景。
    if (print)
    {
        // 限制前缀长度，避免用户输入过长时挤压固定大小的日志缓冲区。
        constexpr std::size_t maxPrefixLength = 200;
        // std::min 同时保证 size_t 到 int 的转换不会超过后续格式参数的合理范围。
        int const prefixLength = static_cast<int>(std::min(prefix.size(), maxPrefixLength));
        // 空前缀传递空 C 字符串；非空时直接借用 string_view 的内存，不复制文本。
        char const* const prefixData = prefix.empty() ? "" : prefix.data();

        // 256 字节足以容纳前缀、格式字符和毫秒数；snprintf 会保证不越界写入。
        char message[256]{};
        // 使用动态精度参数截断前缀，并把耗时格式化为三位小数。
        std::snprintf(message, sizeof(message), "[%.*s]: %.3f ms",
            prefixLength, prefixData, latency);
        // 统一通过 Validator 输出 stdout 并写入每日日志文件。
        Validator::info(message);
    }
    // 返回本次 GPU 操作的毫秒耗时；调用方可自行决定是否打印或聚合。
    return latency;
}

} // namespace trt_timer
