// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

// 把 CUDA 的专用释放函数适配为 unique_ptr deleter，统一管理非 new 创建的资源。
struct CudaDeviceDeleter final
{
    // 释放 cudaMalloc 返回的 device 指针；清理路径不得抛出异常。
    void operator()(void* memory) const noexcept;
};

struct CudaPinnedDeleter final
{
    // 释放 cudaMallocHost 返回的 page-locked host 指针。
    void operator()(void* memory) const noexcept;
};

struct CudaStreamDeleter final
{
    // unique_ptr 的 pointer 类型直接使用 CUDA stream 句柄。
    using pointer = cudaStream_t;

    // 销毁当前 Model 独占的 CUDA stream。
    void operator()(cudaStream_t stream) const noexcept;
};

struct CudaEventDeleter final
{
    // unique_ptr 的 pointer 类型直接使用 CUDA event 句柄。
    using pointer = cudaEvent_t;

    // 销毁 Timer 独占的 CUDA event。
    void operator()(cudaEvent_t event) const noexcept;
};

using CudaDeviceMemory = std::unique_ptr<void, CudaDeviceDeleter>;
using CudaPinnedMemory = std::unique_ptr<void, CudaPinnedDeleter>;
using CudaStream = std::unique_ptr<void, CudaStreamDeleter>;
using CudaEvent = std::unique_ptr<void, CudaEventDeleter>;

// 分配一块正数大小的 device 显存，并交给对应 deleter 管理。
[[nodiscard]] CudaDeviceMemory allocateCudaDevice(std::size_t bytes);
// 分配可用于异步 H2D 的 pinned host 内存，并交给对应 deleter 管理。
[[nodiscard]] CudaPinnedMemory allocateCudaPinned(std::size_t bytes);
// 创建一条非默认 CUDA stream，供一个 Model 的流水线复用。
[[nodiscard]] CudaStream createCudaStream();
// 创建一个 CUDA event，供 Timer 记录指定 stream 上的时间点。
[[nodiscard]] CudaEvent createCudaEvent();
// 在可抛异常路径切换当前线程的 CUDA device。
void selectCudaDevice(int deviceId);
// 在 Model 的 noexcept 清理路径切换当前线程的 CUDA device；失败只记录日志。
void selectCudaDeviceNoexcept(int deviceId) noexcept;
// 在不抛异常的清理路径中等待指定 stream 上的所有工作完成。
void synchronizeCudaStreamNoexcept(cudaStream_t stream) noexcept;
