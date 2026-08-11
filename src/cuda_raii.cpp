// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_raii.h"

#include "validator.h"

#include <cstdio>
#include <string_view>

namespace
{

void logCleanupError(
    cudaError_t const status,
    std::string_view const operation) noexcept
{
    if (status == cudaSuccess)
    {
        return;
    }

    char message[256]{};
    std::snprintf(
        message,
        sizeof(message),
        "%.*s failed: %s",
        static_cast<int>(operation.size()),
        operation.data(),
        cudaGetErrorString(status));
    Validator::log(message);
}

} // namespace

void CudaDeviceDeleter::operator()(void* memory) const noexcept
{
    // unique_ptr 释放空指针也合法；CUDA Runtime 会返回成功或记录清理错误。
    logCleanupError(cudaFree(memory), "cudaFree");
}

void CudaPinnedDeleter::operator()(void* memory) const noexcept
{
    // pinned host workspace 必须使用与分配匹配的 cudaFreeHost。
    logCleanupError(cudaFreeHost(memory), "cudaFreeHost");
}

void CudaStreamDeleter::operator()(cudaStream_t stream) const noexcept
{
    // Model::reset() 已在此之前同步该实例 stream。
    logCleanupError(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

void CudaEventDeleter::operator()(cudaEvent_t event) const noexcept
{
    // Timer 析构时事件不再有未完成的测量依赖。
    logCleanupError(cudaEventDestroy(event), "cudaEventDestroy");
}

CudaDeviceMemory allocateCudaDevice(std::size_t const bytes)
{
    // 调用方按一个 Model 的最大 batch 请求容量，禁止零字节句柄进入运行态。
    Assertf(bytes > 0, "CUDA device allocation size must be positive");

    void* memory = nullptr;
    checkRuntime(cudaMalloc(&memory, bytes));
    return CudaDeviceMemory{memory};
}

CudaPinnedMemory allocateCudaPinned(std::size_t const bytes)
{
    // pinned 内存是 cudaMemcpyAsync(H2D) 使用的 host 端源缓冲区。
    Assertf(bytes > 0, "CUDA pinned allocation size must be positive");

    void* memory = nullptr;
    checkRuntime(cudaMallocHost(&memory, bytes));
    return CudaPinnedMemory{memory};
}

CudaStream createCudaStream()
{
    // 使用非默认 stream，使不同 Model 的提交队列能够被 GPU 独立调度。
    cudaStream_t stream{};
    checkRuntime(cudaStreamCreate(&stream));
    return CudaStream{stream};
}

CudaEvent createCudaEvent()
{
    // 每个 Timer 创建一对独立事件。
    cudaEvent_t event{};
    checkRuntime(cudaEventCreate(&event));
    return CudaEvent{event};
}

void selectCudaDevice(int const deviceId)
{
    // 先查询设备数量，把错误的编号转换成带上下文的项目异常。
    int deviceCount = 0;
    checkRuntime(cudaGetDeviceCount(&deviceCount));
    Assertf(deviceId >= 0 && deviceId < deviceCount,
        "CUDA device %d is outside [0,%d)", deviceId, deviceCount);
    // CUDA 的当前 device 由当前调用显式设置。
    checkRuntime(cudaSetDevice(deviceId));
}

void selectCudaDeviceNoexcept(int const deviceId) noexcept
{
    // reset() 不能抛异常；切换失败时由统一清理日志记录。
    if (deviceId >= 0)
    {
        logCleanupError(cudaSetDevice(deviceId), "cudaSetDevice");
    }
}

void synchronizeCudaStreamNoexcept(cudaStream_t stream) noexcept
{
    // reset/清理路径不能抛异常；错误只写日志并继续释放其余资源。
    if (stream != nullptr)
    {
        logCleanupError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}
