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

// 一个 Model 是一套独立的推理运行时，持有一组 TensorRT 和 CUDA 资源；reset() 会按
// 依赖的逆序释放它们。不同实例可以使用同一份只读 EngineData 初始化，
// 但不会共享下面的 runtime、engine、context、stream 或显存地址。
struct Model
{
    // 默认构造只建立空壳；真正的 TensorRT/CUDA 资源由 initModel() 创建。
    Model() = default;
    // 析构通过 reset() 等待并释放当前实例的全部运行态资源。
    ~Model() noexcept;

    // 句柄和显存属于唯一 owner，禁止复制以避免 double free 或共享可变 context。
    Model(Model const&) = delete;
    Model& operator=(Model const&) = delete;
    // 线程入口捕获 Model&，禁止移动以避免已绑定地址的运行态失去所有者。
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    // 本项目把 context、workspace、buffer 和 stream 作为一个复合运行态；同一实例应由
    // 一个 worker 按顺序使用。需要并发推理时应创建多个 Model 实例，各自传递 Model&。
    void reset() noexcept;

    // 该实例的日志和窗口标签，例如 modelA；由 initModel() 设置。
    std::string name{};
    // 创建和使用该实例的 CUDA device 编号；-1 表示尚未初始化。
    int deviceId{-1};

    // 反序列化 Engine 所依赖的 TensorRT runtime；每个 Model 独占一份。
    std::unique_ptr<nvinfer1::IRuntime> runtime{};
    // 当前实例自己的网络计划和权重对象，不与其他 Model 共享。
    std::unique_ptr<nvinfer1::ICudaEngine> engine{};
    // 当前实例的执行上下文；动态 batch 和 I/O 地址都保存在这里。
    std::unique_ptr<nvinfer1::IExecutionContext> context{};
    // 该实例提交预处理、推理和拷贝操作的 CUDA stream。
    CudaStream stream{};
    // TensorRT 输入张量的 device 缓冲区，按 maxBatch 预分配并复用。
    CudaDeviceMemory inputDevice{};
    // TensorRT 输出张量的 device 缓冲区，按 maxBatch/maxDetections 预分配并复用。
    CudaDeviceMemory outputDevice{};
    // 主机端 pinned workspace，保存 d2i 矩阵和打包后的 BGR 图片。
    CudaPinnedMemory preprocessHost{};
    // device 端预处理 workspace，对应 preprocessHost 的同一批布局。
    CudaDeviceMemory preprocessDevice{};
    // 上述两个 workspace 当前可容纳的字节数；不足时只扩容本实例自己的缓冲区。
    std::size_t preprocessCapacity{};

    // Engine 输入/输出张量的名称，用于 context 的 shape 和地址绑定。
    std::string inputName{};
    std::string outputName{};
    // 当前 context 使用的完整输入 shape，d[0] 会在每轮推理前改成实际 batch。
    nvinfer1::Dims inputShape{};
    // 预处理 kernel 使用的目标输入高度。
    int inputHeight{};
    // 预处理 kernel 使用的目标输入宽度。
    int inputWidth{};
    // 当前 optimization profile 允许的最大 batch。
    int maxBatch{};
    // 每张图片输出张量中允许解析的最大检测框数量。
    int maxDetections{};
};

// 以二进制方式读取 plan 文件；返回的字节数组在所有 initModel 调用结束前保持有效。
[[nodiscard]] EngineData readEngine(std::filesystem::path const& enginePath);
// 用只读 plan 字节初始化一个 Model；函数会先 reset 旧实例，再创建该实例的完整运行态。
// 同一份 engineData 可以顺序传给多个 Model，但每次调用都会独立反序列化和分配资源。
// modelName 会成为该实例及其 worker 日志的前缀；deviceId 会在资源创建前绑定。
void initModel(
    Model& model,
    std::span<char const> engineData,
    std::string_view modelName,
    int deviceId);
