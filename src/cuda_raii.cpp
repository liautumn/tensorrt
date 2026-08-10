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
    logCleanupError(cudaFree(memory), "cudaFree");
}

void CudaPinnedDeleter::operator()(void* memory) const noexcept
{
    logCleanupError(cudaFreeHost(memory), "cudaFreeHost");
}

void CudaStreamDeleter::operator()(cudaStream_t stream) const noexcept
{
    logCleanupError(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

void CudaEventDeleter::operator()(cudaEvent_t event) const noexcept
{
    logCleanupError(cudaEventDestroy(event), "cudaEventDestroy");
}

CudaDeviceMemory allocateCudaDevice(std::size_t const bytes)
{
    Assertf(bytes > 0, "CUDA device allocation size must be positive");

    void* memory = nullptr;
    checkRuntime(cudaMalloc(&memory, bytes));
    return CudaDeviceMemory{memory};
}

CudaPinnedMemory allocateCudaPinned(std::size_t const bytes)
{
    Assertf(bytes > 0, "CUDA pinned allocation size must be positive");

    void* memory = nullptr;
    checkRuntime(cudaMallocHost(&memory, bytes));
    return CudaPinnedMemory{memory};
}

CudaStream createCudaStream()
{
    cudaStream_t stream{};
    checkRuntime(cudaStreamCreate(&stream));
    return CudaStream{stream};
}

CudaEvent createCudaEvent()
{
    cudaEvent_t event{};
    checkRuntime(cudaEventCreate(&event));
    return CudaEvent{event};
}

void synchronizeCudaStreamNoexcept(cudaStream_t stream) noexcept
{
    if (stream != nullptr)
    {
        logCleanupError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}
