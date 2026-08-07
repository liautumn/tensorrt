// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

// Windows 头文件中的 min/max 宏会干扰标准库接口，包含 TensorRT 前先关闭它们。
#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

// TensorRT 11 中 Dims 是 Dims64 的类型别名，必须使用官方声明，不能按 class 前置声明。
#include <NvInfer.h>
// CUDA Runtime 接口提供错误码以及 cudaPeekAtLastError()。
#include <cuda_runtime_api.h>

// std::string 保存自动读取的张量名称，std::string_view 让日志入口无需复制文本。
#include <string>
#include <string_view>

// 集中保存项目使用的 CUDA、普通条件和模型契约校验，不持有业务状态。
class Validator final
{
public:
    // 校验类不保存状态，只通过下面的静态方法提供统一检查。
    Validator() = delete;

    // 检查 CUDA Runtime 返回码，失败时记录调用、错误名称、说明和错误码。
    static void checkCuda(
        cudaError_t status,
        char const* call);
    // 检查普通布尔条件；assertionf 额外支持 printf 风格的上下文信息。
    static void assertion(
        bool condition,
        char const* expression);
    static void assertionf(
        bool condition,
        char const* format,
        ...);
    // 错误日志输出到 stderr，信息日志输出到 stdout；两者同时写入每日文件。
    static void log(std::string_view message) noexcept;
    static void info(std::string_view message) noexcept;

    // 检查本项目实际支持的 TensorRT 输入、输出、类型及动态 shape 契约。
    static void checkModel(
        nvinfer1::ICudaEngine const& engine,
        std::string const& inputName,
        std::string const& outputName);
    // 输入 shape 设置完成后，检查执行上下文解析出的最终输出 shape。
    // expectedDetections 大于 0 时还要求不同 batch 的 max_det 保持不变。
    static void checkOutputShape(
        nvinfer1::Dims const& shape,
        int expectedBatch,
        int expectedDetections = 0);
};

// 单行执行 CUDA API，并确保表达式只被求值一次。
#define checkRuntime(call) \
    do \
    { \
        Validator::checkCuda((call), #call); \
    } while (false)

// 提交 CUDA kernel 后立即检查 launch configuration 等启动错误。
#define checkKernel(...) \
    do \
    { \
        (__VA_ARGS__); \
        checkRuntime(cudaPeekAtLastError()); \
    } while (false)

// 普通条件检查在 Release 构建中也会保留，不使用可能被 NDEBUG 移除的标准 assert。
#define Assert(operation) \
    do \
    { \
        if (!static_cast<bool>(operation)) \
        { \
            Validator::assertion(false, #operation); \
        } \
    } while (false)

// 仅在条件失败时求值格式参数，避免正常路径产生字符串格式化开销或额外副作用。
#define Assertf(operation, ...) \
    do \
    { \
        if (!static_cast<bool>(operation)) \
        { \
            Validator::assertionf(false, __VA_ARGS__); \
        } \
    } while (false)
