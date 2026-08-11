// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "cuda_raii.h"

#include <NvInfer.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

using EngineData = std::vector<char>;

// 按依赖顺序持有 TensorRT 和 CUDA 资源；析构时会按相反顺序自动释放。
struct Model
{
    Model() = default;
    ~Model() noexcept;

    Model(Model const&) = delete;
    Model& operator=(Model const&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    void reset() noexcept;

    std::unique_ptr<nvinfer1::IRuntime> runtime{};
    std::unique_ptr<nvinfer1::ICudaEngine> engine{};
    CudaStream stream{};
    CudaDeviceMemory inputDevice{};
    CudaDeviceMemory outputDevice{};
    CudaPinnedMemory preprocessHost{};
    CudaDeviceMemory preprocessDevice{};
    std::unique_ptr<nvinfer1::IExecutionContext> context{};
    std::size_t preprocessCapacity{};

    std::string inputName{};
    std::string outputName{};
    nvinfer1::Dims inputShape{};
    int inputHeight{};
    int inputWidth{};
    int maxBatch{};
    int maxDetections{};
};

[[nodiscard]] EngineData readEngine(std::filesystem::path const& enginePath);
void initModel(Model& model, std::span<char const> engineData);
