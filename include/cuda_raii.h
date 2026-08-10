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
    void operator()(void* memory) const noexcept;
};

struct CudaPinnedDeleter final
{
    void operator()(void* memory) const noexcept;
};

struct CudaStreamDeleter final
{
    using pointer = cudaStream_t;

    void operator()(cudaStream_t stream) const noexcept;
};

struct CudaEventDeleter final
{
    using pointer = cudaEvent_t;

    void operator()(cudaEvent_t event) const noexcept;
};

using CudaDeviceMemory = std::unique_ptr<void, CudaDeviceDeleter>;
using CudaPinnedMemory = std::unique_ptr<void, CudaPinnedDeleter>;
using CudaStream = std::unique_ptr<void, CudaStreamDeleter>;
using CudaEvent = std::unique_ptr<void, CudaEventDeleter>;

[[nodiscard]] CudaDeviceMemory allocateCudaDevice(std::size_t bytes);
[[nodiscard]] CudaPinnedMemory allocateCudaPinned(std::size_t bytes);
[[nodiscard]] CudaStream createCudaStream();
[[nodiscard]] CudaEvent createCudaEvent();
void synchronizeCudaStreamNoexcept(cudaStream_t stream) noexcept;
