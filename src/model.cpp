// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

// 引入本模块公开的 EngineData、Model 和三个模型操作函数声明。
#include "model.h"

// 提供 std::ifstream，用于以二进制方式读取 Engine 文件。
#include <fstream>
// 提供输出流和字符串流，用于格式化模型输入输出张量信息。
#include <ostream>
#include <sstream>
#include <string_view>
#include <utility>

// 集中使用 Assertf、checkRuntime、模型契约校验和文件日志。
#include "validator.h"

// 匿名命名空间使 Logger 和 logger 只在当前 model.cpp 文件内可见。
namespace
{

// 实现 TensorRT 要求的日志接口，用于接收 TensorRT 初始化和运行期间的日志。
struct Logger final : nvinfer1::ILogger
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
            // 直接转发原始文本，避免在 TensorRT 的 noexcept 回调中分配临时字符串。
            Validator::log(message);
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
        Assertf(tensorName != nullptr, "Tensor index %d has no name", tensorIndex);
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
    Assertf(inputCount == 1 && outputCount == 1,
        "Expected one input and one output, found %d input(s) and %d output(s)", inputCount, outputCount);
}

// 将 TensorRT 数据类型转换为便于阅读的名称。
[[nodiscard]] constexpr std::string_view dataTypeName(nvinfer1::DataType const dataType) noexcept
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
    Validator::info("model io tensors: " + std::to_string(tensorCount));

    for (int tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex)
    {
        char const* tensorName = engine.getIOTensorName(tensorIndex);
        nvinfer1::TensorIOMode const ioMode = engine.getTensorIOMode(tensorName);

        std::ostringstream message;
        message << "  "
                << (ioMode == nvinfer1::TensorIOMode::kINPUT ? "input" : "output")
                << ": name=" << tensorName
                << " dtype=" << dataTypeName(engine.getTensorDataType(tensorName))
                << " engine_shape=";
        printShape(message, engine.getTensorShape(tensorName));

        // 动态输入额外打印第 0 个优化配置的完整形状范围。
        if (ioMode == nvinfer1::TensorIOMode::kINPUT)
        {
            message << " profile_min=";
            printShape(message,
                engine.getProfileShape(tensorName, 0, nvinfer1::OptProfileSelector::kMIN));
            message << " profile_opt=";
            printShape(message,
                engine.getProfileShape(tensorName, 0, nvinfer1::OptProfileSelector::kOPT));
            message << " profile_max=";
            printShape(message,
                engine.getProfileShape(tensorName, 0, nvinfer1::OptProfileSelector::kMAX));
        }
        Validator::info(message.str());
    }
}

// 结束匿名命名空间。
} // namespace

Model::~Model() noexcept
{
    reset();
}

void Model::reset() noexcept
{
    {
        // 清理日志沿用该实例的标签；空壳 Model 使用通用 app 标签。
        LogContext logContext(name.empty() ? std::string_view{"app"} : std::string_view{name});
        // CUDA device 是线程局部状态，释放前切回资源创建时的 device。
        selectCudaDeviceNoexcept(deviceId);
        // 先等待本实例唯一的 stream，确保 kernel、TensorRT enqueue 和异步拷贝都已结束；
        // 随后才能安全释放 context 依赖的显存和 workspace。Model 之间不会等待彼此的 stream。
        synchronizeCudaStreamNoexcept(stream.get());
        // 按“使用者先于被使用资源”逆序销毁：context -> buffers -> stream -> engine -> runtime。
        context.reset();
        preprocessDevice.reset();
        preprocessHost.reset();
        outputDevice.reset();
        inputDevice.reset();
        stream.reset();
        engine.reset();
        runtime.reset();
    }

    // 资源释放后清空派生容量和描述信息，避免复用对象时残留上一次模型状态。
    preprocessCapacity = 0;
    name.clear();
    deviceId = -1;
    inputName.clear();
    outputName.clear();
    inputShape = {};
    inputHeight = 0;
    inputWidth = 0;
    maxBatch = 0;
    maxDetections = 0;
}

// 读取指定路径的 TensorRT Engine 二进制文件。
// 参数 enginePath：Engine 文件路径，只读传入。
// 返回值：保存文件全部字节的 EngineData。
EngineData readEngine(std::filesystem::path const& enginePath)
{
    std::string const pathText = enginePath.string();
    // 以二进制模式打开文件，并让初始读取位置停在文件末尾，以便直接取得文件大小。
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    // 检查文件是否成功打开。
    Assertf(file, "Cannot open engine: %s", pathText.c_str());

    // file.tellg() 返回当前位置，即文件字节数；据此一次性创建足够大的字节数组。
    auto const engineSize = file.tellg();
    Assertf(engineSize > std::streampos(0), "TensorRT engine is empty: %s", pathText.c_str());
    EngineData engineData(static_cast<std::size_t>(engineSize));
    // 将文件读取位置从末尾移回第 0 个字节，为读取完整文件做准备。
    file.seekg(0);
    // 从文件读取 engineData.size() 个字节，并写入 engineData 的连续内存。
    file.read(engineData.data(), static_cast<std::streamsize>(engineData.size()));
    Assertf(file.gcount() == static_cast<std::streamsize>(engineData.size()),
        "TensorRT engine was not read completely: %s", pathText.c_str());
    // 返回已经装入内存的 Engine 二进制数据。
    return engineData;
}

// 把 Engine 二进制数据初始化成可执行、可重复使用的 TensorRT 模型。
// 参数 engineData：Engine 文件的完整字节内容，只读传入。
// 参数 model：要初始化的模型；函数会先释放其中已有的模型资源。
// 参数 modelName/deviceId：该实例的日志标签和 CUDA 设备编号。
void initModel(
    Model& model,
    std::span<char const> const engineData,
    std::string_view const modelName,
    int const deviceId)
{
    // 先复制名称，保证后续异常路径中的 Model 标签拥有稳定存储。
    std::string requestedName(modelName);
    Assertf(!requestedName.empty(), "Model name must not be empty");
    model.reset();
    Assertf(!engineData.empty(), "TensorRT engine data is empty");
    // 先把标签保存到 Model，再让 LogContext 借用稳定的成员存储。
    model.name = std::move(requestedName);
    model.deviceId = deviceId;
    // 初始化过程中的 TensorRT/CUDA 日志都标记为目标实例。
    LogContext logContext(model.name);
    // 在创建 runtime、stream 和显存前固定当前线程的目标 CUDA device。
    selectCudaDevice(deviceId);
    // 提前触发 CUDA Runtime 初始化，驱动或设备不可用时在反序列化模型前报告。
    checkRuntime(cudaFree(nullptr));
    // 使用本文件的 logger 创建 TensorRT Runtime；Runtime 负责反序列化 Engine。
    model.runtime.reset(nvinfer1::createInferRuntime(logger));
    Assertf(model.runtime != nullptr, "Failed to create TensorRT runtime");
    // 将 engineData.data() 指向的 engineData.size() 个字节反序列化为 CUDA Engine。
    model.engine.reset(model.runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    Assertf(model.engine != nullptr,
        "Engine is corrupted or incompatible with the current TensorRT, CUDA, GPU, or operating system");
    // 枚举 Engine 自带的 I/O 信息，自动保存真实输入和输出张量名称。
    readModelIoNames(*model.engine, model);
    // 在创建 context 和分配显存前集中拒绝类型、位置、格式或 shape 不兼容的模型。
    Validator::checkModel(*model.engine, model.inputName, model.outputName);
    // 从 Engine 创建执行上下文；后续输入形状设置和 enqueueV3 推理都通过它完成。
    model.context.reset(model.engine->createExecutionContext());
    Assertf(model.context != nullptr, "Failed to create TensorRT execution context");

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
    model.stream = createCudaStream();
    // 在 model.stream 上选择编号为 0 的优化配置，使 context 使用上面查询的同一套形状范围。
    Assertf(model.context->setOptimizationProfileAsync(0, model.stream.get()),
        "Failed to set TensorRT optimization profile 0");

    // 暂时把当前输入形状的 batch 设为最大值，用最大输出形状计算和分配显存。
    model.inputShape.d[0] = model.maxBatch;
    // 告诉执行上下文输入张量名称及其当前完整形状 [maxBatch, C, H, W]。
    Assertf(model.context->setInputShape(model.inputName.c_str(), model.inputShape),
        "Failed to set input shape for tensor '%s'", model.inputName.c_str());
    Validator::checkOutputShape(model.context->getTensorShape(model.outputName.c_str()), model.maxBatch);
    // 查询当前输入形状对应的输出形状，并读取 d[1] 作为每张图片的最大检测框数。
    model.maxDetections
        // outputName 指定要查询的输出张量；预期输出布局为 [batch, maxDetections, 6]。
        = static_cast<int>(model.context->getTensorShape(model.outputName.c_str()).d[1]);

    // 初始化阶段枚举并打印模型的全部输入输出，以及输入优化配置的形状范围。
    printModelIo(*model.engine);

    // 为最大 batch 的输入分配 GPU 显存；每张图有 3 个通道、H*W 个 float 元素。
    model.inputDevice = allocateCudaDevice(
        static_cast<std::size_t>(model.maxBatch) * 3 * model.inputHeight * model.inputWidth * sizeof(float));
    // 为最大 batch 的输出分配 GPU 显存；每个检测框由 6 个 float 数值组成。
    model.outputDevice = allocateCudaDevice(
        static_cast<std::size_t>(model.maxBatch) * model.maxDetections * 6 * sizeof(float));

    // 将输入张量名称绑定到输入显存，enqueueV3 时 TensorRT 会从该地址读取图片数据。
    Assertf(model.context->setTensorAddress(model.inputName.c_str(), model.inputDevice.get()),
        "Failed to bind input tensor '%s'", model.inputName.c_str());
    // 将输出张量名称绑定到输出显存，enqueueV3 时 TensorRT 会把检测结果写到该地址。
    Assertf(model.context->setTensorAddress(model.outputName.c_str(), model.outputDevice.get()),
        "Failed to bind output tensor '%s'", model.outputName.c_str());
}
