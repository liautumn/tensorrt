// 引入 Validator 及 checkRuntime、checkKernel、Assert、Assertf 的声明。
#include "validator.h"

// 标准库分别提供可变参数、时间、文件、线程同步、格式化和异常支持。
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace
{

// 把宏捕获的源码文件和行号拼成 file:line，统一用于错误信息。
std::string location(char const* file, int line)
{
    return std::string(file) + ':' + std::to_string(line);
}

// 返回当前可执行文件所在目录；日志位置不依赖进程的工作目录。
std::filesystem::path executableDirectory()
{
#if defined(_WIN32)
    std::wstring path(32768, L'\0');
    DWORD const length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length > 0 && length < path.size())
    {
        path.resize(length);
        return std::filesystem::path(path).parent_path();
    }
#elif defined(__linux__)
    std::error_code error;
    auto const executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error)
    {
        return executable.parent_path();
    }
#endif
    return std::filesystem::current_path();
}

// Windows 与 POSIX 的线程安全本地时间接口名称和参数顺序不同，在此统一。
std::tm localTime(std::time_t value)
{
    std::tm result{};
#if defined(_WIN32)
    localtime_s(&result, &value);
#else
    localtime_r(&value, &result);
#endif
    return result;
}

// 每次写日志都重新计算日期，跨过零点后会自动切换到新的 yyyy-MM-dd.log。
void writeDailyLog(char const* level, std::string_view message) noexcept
{
    try
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        auto const now = std::chrono::system_clock::now();
        std::time_t const time = std::chrono::system_clock::to_time_t(now);
        std::tm const local = localTime(time);

        char date[11]{};
        char timestamp[20]{};
        std::strftime(date, sizeof(date), "%Y-%m-%d", &local);
        std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &local);

        // 路径固定为“可执行文件目录/logs”，不受程序启动时的工作目录影响。
        auto const directory = executableDirectory() / "logs";
        std::filesystem::create_directories(directory);
        std::ofstream output(directory / (std::string(date) + ".log"), std::ios::app);
        if (output)
        {
            output << timestamp << " [" << level << "] " << message << '\n';
        }
    }
    catch (...)
    {
        // 日志失败不能覆盖真正需要报告的模型或 CUDA 错误。
    }
}

// 将 TensorRT shape 转成便于出现在异常信息中的 [N,C,H,W] 格式。
std::string shapeText(nvinfer1::Dims const& shape)
{
    if (shape.nbDims < 0)
    {
        return "invalid";
    }

    std::ostringstream output;
    output << '[';
    for (int index = 0; index < shape.nbDims; ++index)
    {
        if (index != 0)
        {
            output << ',';
        }
        output << shape.d[index];
    }
    output << ']';
    return output.str();
}

// 校验 profile 中一个输入 shape 能安全映射到项目使用的 int 型 NCHW 字段。
void checkInputShape(nvinfer1::Dims const& shape, char const* selector)
{
    Assertf(shape.nbDims == 4, "Input %s shape must be [N,3,H,W], got %s",
        selector, shapeText(shape).c_str());
    Assertf(shape.d[0] > 0 && shape.d[1] == 3 && shape.d[2] > 0 && shape.d[3] > 0,
        "Input %s shape must contain positive N/H/W and C=3, got %s",
        selector, shapeText(shape).c_str());
    auto constexpr maxInt = std::numeric_limits<int>::max();
    Assertf(shape.d[0] <= maxInt && shape.d[2] <= maxInt && shape.d[3] <= maxInt,
        "Input %s shape exceeds the supported int range: %s",
        selector, shapeText(shape).c_str());
    Assertf(shape.d[2] <= maxInt / 3 / shape.d[3],
        "Input %s shape exceeds the CUDA kernel index range: %s",
        selector, shapeText(shape).c_str());
}

} // namespace

// CUDA 失败统一记录 API、错误名称、说明、数值错误码及调用位置，然后抛出异常。
void Validator::checkCuda(
    cudaError_t status,
    char const* call,
    char const* file,
    int line)
{
    if (status == cudaSuccess)
    {
        return;
    }

    std::string const message
        = "CUDA Runtime error at " + location(file, line) + ": " + call
        + " -> " + cudaGetErrorName(status)
        + " (" + std::to_string(static_cast<int>(status)) + "): "
        + cudaGetErrorString(status);
    log(message);
    throw std::runtime_error(message);
}

// 普通断言失败时记录原始条件表达式和调用位置。
void Validator::assertion(
    bool condition,
    char const* expression,
    char const* file,
    int line)
{
    if (!condition)
    {
        std::string const message
            = "Assert failed at " + location(file, line) + ": " + expression;
        log(message);
        throw std::runtime_error(message);
    }
}

// 带格式的断言只在失败路径生成补充上下文，正常路径由宏直接跳过。
void Validator::assertionf(
    bool condition,
    char const* expression,
    char const* file,
    int line,
    char const* format,
    ...)
{
    if (condition)
    {
        return;
    }

    char message[2048]{};
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);

    std::string const error
        = "Assert failed at " + location(file, line) + ": " + expression + " - " + message;
    log(error);
    throw std::runtime_error(error);
}

// 日志入口使用 string_view，TensorRT noexcept 回调和析构路径无需构造临时字符串。
void Validator::log(std::string_view message) noexcept
{
    if (!message.empty())
    {
        std::fwrite(message.data(), sizeof(char), message.size(), stderr);
    }
    std::fputc('\n', stderr);
    writeDailyLog("ERROR", message);
}

void Validator::info(std::string_view message) noexcept
{
    if (!message.empty())
    {
        std::fwrite(message.data(), sizeof(char), message.size(), stdout);
    }
    std::fputc('\n', stdout);
    writeDailyLog("INFO", message);
}

// 校验 Engine 与当前 FP32 NCHW 输入、线性 Device I/O 和 YOLO 输出解析方式兼容。
void Validator::checkModel(
    nvinfer1::ICudaEngine const& engine,
    std::string const& inputName,
    std::string const& outputName)
{
    Assertf(!inputName.empty() && !outputName.empty(), "Model input or output tensor name is empty");
    Assertf(engine.getTensorIOMode(inputName.c_str()) == nvinfer1::TensorIOMode::kINPUT,
        "Tensor '%s' is not an input", inputName.c_str());
    Assertf(engine.getTensorIOMode(outputName.c_str()) == nvinfer1::TensorIOMode::kOUTPUT,
        "Tensor '%s' is not an output", outputName.c_str());
    Assertf(engine.getTensorDataType(inputName.c_str()) == nvinfer1::DataType::kFLOAT,
        "Input tensor '%s' must use FP32", inputName.c_str());
    Assertf(engine.getTensorDataType(outputName.c_str()) == nvinfer1::DataType::kFLOAT,
        "Output tensor '%s' must use FP32", outputName.c_str());
    Assertf(engine.getTensorLocation(inputName.c_str()) == nvinfer1::TensorLocation::kDEVICE,
        "Input tensor '%s' must use device memory", inputName.c_str());
    Assertf(engine.getTensorLocation(outputName.c_str()) == nvinfer1::TensorLocation::kDEVICE,
        "Output tensor '%s' must use device memory", outputName.c_str());
    Assertf(engine.getNbOptimizationProfiles() > 0, "Model has no optimization profile");
    Assertf(engine.getTensorFormat(inputName.c_str(), 0) == nvinfer1::TensorFormat::kLINEAR,
        "Input tensor '%s' must use linear format", inputName.c_str());
    Assertf(engine.getTensorFormat(outputName.c_str(), 0) == nvinfer1::TensorFormat::kLINEAR,
        "Output tensor '%s' must use linear format", outputName.c_str());

    auto const minShape = engine.getProfileShape(
        inputName.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
    auto const optShape = engine.getProfileShape(
        inputName.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
    auto const maxShape = engine.getProfileShape(
        inputName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    checkInputShape(minShape, "minimum");
    checkInputShape(optShape, "optimum");
    checkInputShape(maxShape, "maximum");
    Assertf(minShape.d[0] == 1, "Minimum batch size must be 1, got %lld",
        static_cast<long long>(minShape.d[0]));

    for (int index = 0; index < 4; ++index)
    {
        Assertf(minShape.d[index] <= optShape.d[index]
                && optShape.d[index] <= maxShape.d[index],
            "Input profile must satisfy min <= opt <= max");
    }

    // 按 max batch 和实际使用的 opt H/W 组合校验输入显存字节数不会溢出 size_t。
    auto const maxSize = std::numeric_limits<std::size_t>::max();
    Assertf(static_cast<std::size_t>(maxShape.d[0])
            <= maxSize / sizeof(float) / 3
                / static_cast<std::size_t>(optShape.d[2])
                / static_cast<std::size_t>(optShape.d[3]),
        "Input tensor byte size exceeds the supported size_t range");

    auto const outputShape = engine.getTensorShape(outputName.c_str());
    // Engine 层允许运行期维度为 -1；setInputShape 后会对 context 的实际值做严格检查。
    Assertf(outputShape.nbDims == 3
            && (outputShape.d[0] == -1 || outputShape.d[0] > 0)
            && (outputShape.d[1] == -1 || outputShape.d[1] > 0)
            && (outputShape.d[2] == -1 || outputShape.d[2] == 6),
        "Output shape must be [N,max_det,6], got %s", shapeText(outputShape).c_str());
}

// 校验 context 已解析的输出，避免按错误的 batch、候选框数量或字段数分配和复制显存。
void Validator::checkOutputShape(
    nvinfer1::Dims const& shape,
    int expectedBatch,
    int expectedDetections)
{
    auto constexpr maxInt = std::numeric_limits<int>::max();
    Assertf(shape.nbDims == 3
            && expectedBatch > 0
            && shape.d[0] == expectedBatch
            && shape.d[1] > 0
            && shape.d[1] <= maxInt
            && shape.d[2] == 6,
        "Resolved output shape must be [N,max_det,6], got %s", shapeText(shape).c_str());
    Assertf(static_cast<std::size_t>(expectedBatch)
            <= std::numeric_limits<std::size_t>::max() / sizeof(float) / 6
                / static_cast<std::size_t>(shape.d[1]),
        "Output tensor byte size exceeds the supported size_t range");
    Assertf(expectedDetections <= 0 || shape.d[1] == expectedDetections,
        "Resolved max_det changed from %d to %lld",
        expectedDetections, static_cast<long long>(shape.d[1]));
}
