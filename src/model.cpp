// 引入本模块公开的 EngineData、Model 和三个模型操作函数声明。
#include "model.h"

// 提供 fprintf() 和 stderr，用于输出 TensorRT 日志。
#include <cstdio>
// 提供 std::ifstream，用于以二进制方式读取 Engine 文件。
#include <fstream>
// 提供 std::cout，用于在模型初始化完成后打印输入输出张量信息。
#include <iostream>
// 提供 std::runtime_error，用于在 Engine 文件无法打开时报告错误。
#include <stdexcept>

// 匿名命名空间使 Logger 和 logger 只在当前 model.cpp 文件内可见。
namespace
{

// 实现 TensorRT 要求的日志接口，用于接收 TensorRT 初始化和运行期间的日志。
struct Logger : nvinfer1::ILogger
{
    // TensorRT 产生一条日志时会调用此函数。
    // 参数 severity：日志级别，例如内部错误、错误、警告、信息或详细信息。
    // 参数 message：TensorRT 生成的以 '\0' 结尾的日志文本，只读传入。
    // noexcept：保证日志函数不会向 TensorRT 抛出 C++ 异常。
    // 返回值：无。
    void log(Severity severity, char const* message) noexcept override
    {
        // 只保留警告及更严重的日志，忽略普通信息和详细调试日志。
        if (severity <= Severity::kWARNING)
        {
            // 将日志文本写到标准错误流，并在末尾补一个换行符。
            fprintf(stderr, "%s\n", message);
        }
    }
};

// 创建一个当前源文件共用的日志对象，传给 TensorRT Runtime。
Logger logger;

// 从 Engine 自动读取单个输入和单个输出的张量名称。
void readModelIoNames(nvinfer1::ICudaEngine const& engine, Model& model)
{
    int inputCount = 0;
    int outputCount = 0;

    for (int tensorIndex = 0; tensorIndex < engine.getNbIOTensors(); ++tensorIndex)
    {
        char const* tensorName = engine.getIOTensorName(tensorIndex);
        nvinfer1::TensorIOMode const ioMode = engine.getTensorIOMode(tensorName);

        if (ioMode == nvinfer1::TensorIOMode::kINPUT)
        {
            ++inputCount;
            model.inputName = tensorName;
        }
        else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            ++outputCount;
            model.outputName = tensorName;
        }
    }

    // 当前预处理、显存分配和后处理布局只支持单输入、单输出模型。
    if (inputCount != 1 || outputCount != 1)
    {
        throw std::runtime_error("Expected exactly one input and one output tensor, but found "
            + std::to_string(inputCount) + " input(s) and " + std::to_string(outputCount) + " output(s)");
    }
}

// 将 TensorRT 数据类型转换为便于阅读的名称。
char const* dataTypeName(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        return "FP32";
    case nvinfer1::DataType::kHALF:
        return "FP16";
    case nvinfer1::DataType::kINT8:
        return "INT8";
    case nvinfer1::DataType::kINT32:
        return "INT32";
    case nvinfer1::DataType::kBOOL:
        return "BOOL";
    case nvinfer1::DataType::kUINT8:
        return "UINT8";
    case nvinfer1::DataType::kFP8:
        return "FP8";
    case nvinfer1::DataType::kBF16:
        return "BF16";
    case nvinfer1::DataType::kINT64:
        return "INT64";
    case nvinfer1::DataType::kINT4:
        return "INT4";
    case nvinfer1::DataType::kFP4:
        return "FP4";
    case nvinfer1::DataType::kE8M0:
        return "E8M0";
    }
    return "UNKNOWN";
}

// 按 [N,C,H,W] 这种形式把任意维数的 TensorRT shape 写入输出流。
void printShape(std::ostream& output, nvinfer1::Dims const& shape)
{
    output << '[';
    for (int dimensionIndex = 0; dimensionIndex < shape.nbDims; ++dimensionIndex)
    {
        if (dimensionIndex != 0)
        {
            output << ',';
        }
        output << shape.d[dimensionIndex];
    }
    output << ']';
}

// 枚举 Engine 中的全部 I/O 张量，并打印方向、名称、数据类型和形状。
void printModelIo(nvinfer1::ICudaEngine const& engine)
{
    int const tensorCount = engine.getNbIOTensors();
    std::cout << "model io tensors: " << tensorCount << '\n';

    for (int tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex)
    {
        char const* tensorName = engine.getIOTensorName(tensorIndex);
        nvinfer1::TensorIOMode const ioMode = engine.getTensorIOMode(tensorName);

        std::cout << "  "
                  << (ioMode == nvinfer1::TensorIOMode::kINPUT ? "input" : "output")
                  << ": name=" << tensorName
                  << " dtype=" << dataTypeName(engine.getTensorDataType(tensorName))
                  << " engine_shape=";
        printShape(std::cout, engine.getTensorShape(tensorName));

        // 动态输入额外打印第 0 个优化配置的完整形状范围。
        if (ioMode == nvinfer1::TensorIOMode::kINPUT)
        {
            std::cout << " profile_min=";
            printShape(std::cout,
                engine.getProfileShape(tensorName, 0, nvinfer1::OptProfileSelector::kMIN));
            std::cout << " profile_opt=";
            printShape(std::cout,
                engine.getProfileShape(tensorName, 0, nvinfer1::OptProfileSelector::kOPT));
            std::cout << " profile_max=";
            printShape(std::cout,
                engine.getProfileShape(tensorName, 0, nvinfer1::OptProfileSelector::kMAX));
        }
        std::cout << '\n';
    }
}

// 结束匿名命名空间。
} // namespace

// 读取指定路径的 TensorRT Engine 二进制文件。
// 参数 enginePath：Engine 文件路径，只读传入。
// 返回值：保存文件全部字节的 EngineData。
EngineData readEngine(std::string const& enginePath)
{
    // 以二进制模式打开文件，并让初始读取位置停在文件末尾，以便直接取得文件大小。
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    // 检查文件是否成功打开。
    if (!file)
    {
        // 打开失败时终止当前流程，并在异常信息中带上失败的文件路径。
        throw std::runtime_error("Cannot open engine: " + enginePath);
    }

    // file.tellg() 返回当前位置，即文件字节数；据此一次性创建足够大的字节数组。
    EngineData engineData(static_cast<std::size_t>(file.tellg()));
    // 将文件读取位置从末尾移回第 0 个字节，为读取完整文件做准备。
    file.seekg(0);
    // 从文件读取 engineData.size() 个字节，并写入 engineData 的连续内存。
    file.read(engineData.data(), static_cast<std::streamsize>(engineData.size()));
    // 返回已经装入内存的 Engine 二进制数据。
    return engineData;
}

// 把 Engine 二进制数据初始化成可执行、可重复使用的 TensorRT 模型。
// 参数 engineData：Engine 文件的完整字节内容，只读传入。
// 返回值：持有 TensorRT 对象、CUDA 流、输入输出显存及模型尺寸信息的 Model。
Model initModel(EngineData const& engineData)
{
    // 创建一个字段均为默认初始值的 Model，随后逐项填充所需资源和尺寸。
    Model model;
    // 使用本文件的 logger 创建 TensorRT Runtime；Runtime 负责反序列化 Engine。
    model.runtime = nvinfer1::createInferRuntime(logger);
    // 将 engineData.data() 指向的 engineData.size() 个字节反序列化为 CUDA Engine。
    model.engine = model.runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    // 枚举 Engine 自带的 I/O 信息，自动保存真实输入和输出张量名称。
    readModelIoNames(*model.engine, model);
    // 从 Engine 创建执行上下文；后续输入形状设置和 enqueueV3 推理都通过它完成。
    model.context = model.engine->createExecutionContext();

    // 读取第 0 个优化配置中输入张量的最优形状；kOPT 是 Engine 重点优化的常用形状。
    auto const optShape = model.engine->getProfileShape(
        // 第 1 个参数是输入张量名称，第 2 个参数 0 是优化配置编号，第 3 个参数指定读取 kOPT 形状。
        model.inputName.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
    // 读取第 0 个优化配置中输入张量允许的最大形状，用于确定最大 batch。
    auto const maxShape = model.engine->getProfileShape(
        // 第 1 个参数是输入张量名称，第 2 个参数 0 是优化配置编号，第 3 个参数指定读取 kMAX 形状。
        model.inputName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);

    // 把最优输入形状保存为模型当前输入形状的基础值，实际推理时只需修改 batch 维。
    model.inputShape = optShape;
    // 输入通常是 [N, C, H, W]，d[2] 是 H；转成 int 后保存为预处理目标高度。
    model.inputHeight = static_cast<int>(optShape.d[2]);
    // 输入通常是 [N, C, H, W]，d[3] 是 W；转成 int 后保存为预处理目标宽度。
    model.inputWidth = static_cast<int>(optShape.d[3]);
    // 最大形状的 d[0] 是 N，也就是 Engine 单轮推理允许的最大图片数量。
    model.maxBatch = static_cast<int>(maxShape.d[0]);

    // 创建一条 CUDA 流，并把创建结果写入 model.stream，供后续推理过程持续复用。
    cudaStreamCreate(&model.stream);
    // 在 model.stream 上选择编号为 0 的优化配置，使 context 使用上面查询的同一套形状范围。
    model.context->setOptimizationProfileAsync(0, model.stream);

    // 暂时把当前输入形状的 batch 设为最大值，用最大输出形状计算和分配显存。
    model.inputShape.d[0] = model.maxBatch;
    // 告诉执行上下文输入张量名称及其当前完整形状 [maxBatch, C, H, W]。
    model.context->setInputShape(model.inputName.c_str(), model.inputShape);
    // 查询当前输入形状对应的输出形状，并读取 d[1] 作为每张图片的最大检测框数。
    model.maxDetections
        // outputName 指定要查询的输出张量；预期输出布局为 [batch, maxDetections, 6]。
        = static_cast<int>(model.context->getTensorShape(model.outputName.c_str()).d[1]);

    // 初始化阶段枚举并打印模型的全部输入输出，以及输入优化配置的形状范围。
    printModelIo(*model.engine);

    // 为最大 batch 的输入分配 GPU 显存；每张图有 3 个通道、H*W 个 float 元素。
    cudaMalloc(&model.inputDevice,
        // 总字节数 = 最大图片数 * 3 通道 * 输入高度 * 输入宽度 * 单个 float 的字节数。
        static_cast<std::size_t>(model.maxBatch) * 3 * model.inputHeight * model.inputWidth * sizeof(float));
    // 为最大 batch 的输出分配 GPU 显存；每个检测框由 6 个 float 数值组成。
    cudaMalloc(&model.outputDevice,
        // 总字节数 = 最大图片数 * 每张图最大检测框数 * 每个检测框 6 个值 * float 字节数。
        static_cast<std::size_t>(model.maxBatch) * model.maxDetections * 6 * sizeof(float));

    // 将输入张量名称绑定到输入显存，enqueueV3 时 TensorRT 会从该地址读取图片数据。
    model.context->setTensorAddress(model.inputName.c_str(), model.inputDevice);
    // 将输出张量名称绑定到输出显存，enqueueV3 时 TensorRT 会把检测结果写到该地址。
    model.context->setTensorAddress(model.outputName.c_str(), model.outputDevice);
    // 返回初始化完成的模型；调用者可重复使用其中的 context、流和显存执行多轮推理。
    return model;
}

// 释放 Model 持有的 CUDA 和 TensorRT 资源。
// 参数 model：initModel() 返回的 Model，以可修改引用传入。
// 返回值：无。
void releaseModel(Model& model)
{
    // 只有执行过 CUDA 预处理时 workspace 才会被懒分配；空指针表示没有资源需要释放。
    if (model.preprocessDevice != nullptr)
    {
        // 释放为最近批次原图和 d2i 矩阵复用的 GPU workspace。
        cudaFree(model.preprocessDevice);
    }
    if (model.preprocessHost != nullptr)
    {
        // 释放与 GPU workspace 同布局、供 cudaMemcpyAsync 使用的 pinned CPU workspace。
        cudaFreeHost(model.preprocessHost);
    }
    // 清空已释放的设备地址，避免 Model 中保留悬空指针。
    model.preprocessDevice = nullptr;
    // 清空已释放的 pinned host 地址。
    model.preprocessHost = nullptr;
    // 两块 workspace 都已释放，对应的可复用容量恢复为 0。
    model.preprocessCapacity = 0;
    // 释放保存模型输入数据的 GPU 显存。
    cudaFree(model.inputDevice);
    // 释放保存模型输出数据的 GPU 显存。
    cudaFree(model.outputDevice);
    // 销毁初始化时创建并在各轮推理中复用的 CUDA 流。
    cudaStreamDestroy(model.stream);
    // 销毁执行上下文，释放它维护的输入形状和执行状态。
    delete model.context;
    // 销毁反序列化得到的 CUDA Engine。
    delete model.engine;
    // 最后销毁创建 Engine 的 TensorRT Runtime。
    delete model.runtime;
}
