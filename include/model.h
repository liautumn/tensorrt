// 防止同一个头文件在一次编译过程中被重复包含。
#pragma once

// TensorRT 核心接口：提供 IRuntime、ICudaEngine、IExecutionContext、Dims 等类型。
#include <NvInfer.h>
// CUDA Runtime 接口：提供 cudaStream_t、cudaMalloc、cudaFree 等声明。
#include <cuda_runtime_api.h>

// 提供 std::string，用来保存 Engine 文件路径和模型输入/输出张量名称。
#include <string>
// 提供 std::size_t，用来记录 CUDA 预处理 workspace 的字节容量。
#include <cstddef>
// 提供 std::vector，用来保存从磁盘读取的 Engine 二进制数据。
#include <vector>

// EngineData 表示完整的 TensorRT Engine 二进制数据；每个 char 保存一个原始字节。
using EngineData = std::vector<char>;

// Model 集中保存一次模型初始化后需要长期复用的 TensorRT 和 CUDA 资源。
struct Model
{
    // TensorRT Runtime：负责把 Engine 二进制数据反序列化为可使用的模型。
    nvinfer1::IRuntime* runtime{};
    // TensorRT Engine：保存已构建模型的网络结构、权重和优化配置。
    nvinfer1::ICudaEngine* engine{};
    // TensorRT 执行上下文：保存本次运行使用的输入形状，并负责提交推理。
    nvinfer1::IExecutionContext* context{};
    // CUDA 流：让设置优化配置、数据传输和模型推理按照同一条任务流执行。
    cudaStream_t stream{};
    // 输入显存地址：保存预处理后、即将传给模型的 NCHW float 图片数据。
    void* inputDevice{};
    // 输出显存地址：保存模型在 GPU 上生成的检测结果。
    void* outputDevice{};
    // CUDA 预处理懒分配的 pinned CPU workspace；布局为 d2i 表、对齐区、各图紧密 BGR。
    // 它由 cudaMallocHost 创建，不会被换出，可作为 cudaMemcpyAsync 的异步 H2D 源地址。
    void* preprocessHost{};
    // 与 preprocessHost 字节布局相同的 GPU workspace，接收紧密排列的 uint8 BGR 原图和 d2i。
    // kernel 从这里读取原图，转换后的 FP32 NCHW 数据则直接写入 inputDevice。
    void* preprocessDevice{};
    // 每一块 workspace 当前各自拥有的字节容量；容量不足时扩展，足够时跨轮复用。
    std::size_t preprocessCapacity{};

    // 模型输入张量名称；默认名称是 images，必须与导出 Engine 时的名称一致。
    std::string inputName{"images"};
    // 模型输出张量名称；默认名称是 output0，必须与导出 Engine 时的名称一致。
    std::string outputName{"output0"};
    // 当前使用的输入张量形状，通常按 [batch, channel, height, width] 排列。
    nvinfer1::Dims inputShape{};
    // 模型要求的输入图片高度，对应 inputShape.d[2]。
    int inputHeight{};
    // 模型要求的输入图片宽度，对应 inputShape.d[3]。
    int inputWidth{};
    // Engine 优化配置允许的一次推理最大图片数量，对应最大输入形状的 batch 维。
    int maxBatch{};
    // 模型为每张图片预留的最大检测框数量，对应输出形状的第 2 个维度。
    int maxDetections{};
};

// 从磁盘读取 TensorRT Engine 文件。
// 参数 enginePath：Engine 文件路径，例如 "model.engine"。
// 返回值：包含 Engine 文件全部原始字节的 EngineData，供 initModel() 反序列化。
EngineData readEngine(std::string const& enginePath);
// 根据 Engine 二进制数据创建并初始化可重复推理的模型资源。
// 参数 engineData：readEngine() 返回的 Engine 原始字节，只读传入，不会被修改。
// 返回值：初始化完成的 Model，包含 runtime、engine、context、CUDA 流和输入输出显存。
Model initModel(EngineData const& engineData);
// 释放 initModel() 创建的全部 GPU 和 TensorRT 资源。
// 参数 model：要释放的模型对象，以引用传入，因此函数操作的是调用者持有的对象。
// 返回值：无。
void releaseModel(Model& model);
