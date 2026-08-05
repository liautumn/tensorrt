// 引入 TensorRT C++ 推理接口，类似 Java 中 import TensorRT 相关类。
#include <NvInfer.h>
// 引入 CUDA Runtime 接口，用来创建 stream、申请显存和复制数据。
#include <cuda_runtime_api.h>
// 引入 OpenCV 图片读取接口，主要使用 cv::imread。
#include <opencv2/imgcodecs.hpp>
// 引入 OpenCV 图片处理接口，主要使用 resize、cvtColor 和 split。
#include <opencv2/imgproc.hpp>

// 提供 min、max、clamp 和 copy_n 等通用算法。
#include <algorithm>
// 提供 round 等数学函数。
#include <cmath>
// 提供 int32_t、int64_t 等长度固定的整数类型。
#include <cstdint>
// 提供 memcpy，用来读取 Ultralytics engine 的元数据长度。
#include <cstring>
// 提供 ifstream，用二进制方式读取 engine 文件。
#include <fstream>
// 提供 setprecision，用来控制置信度和坐标的打印精度。
#include <iomanip>
// 提供 cout 和 cerr，类似 Java 的 System.out 和 System.err。
#include <iostream>
// 提供 unique_ptr，作用类似自动释放资源的独占引用。
#include <memory>
// 提供 runtime_error，类似 Java 的 RuntimeException。
#include <stdexcept>
// 提供 std::string，作用类似 Java String。
#include <string>
// 提供连续内存数组 std::vector，功能上可类比 Java ArrayList。
#include <vector>

// 编译时检查 TensorRT 主版本，低于 11 就直接停止编译。
#if NV_TENSORRT_MAJOR < 11
#error "This example requires TensorRT 11 or newer."
#endif

// 匿名命名空间让本文件中的辅助类和函数不会暴露给其他 cpp 文件。
namespace
{

// 端到端输出已经完成框解码和 Top-K，这里只过滤低置信度候选框。
// constexpr 表示编译期常量，类似 Java 的 static final。
constexpr float kConfidenceThreshold = 0.25F;

// TensorRT 日志器：只输出警告和错误，避免正常运行时日志过多。
// final 表示不允许再继承；“: public ILogger”相当于 Java implements ILogger。
class Logger final : public nvinfer1::ILogger
{
public:
    // override 表示重写父接口方法；noexcept 表示该函数不会向外抛异常。
    void log(Severity severity, char const* message) noexcept override
    {
        // TensorRT 严重等级数值越小越严重，只保留 warning、error 和 internal error。
        if (severity <= Severity::kWARNING)
        {
            // 把 TensorRT 日志写到标准错误流。
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

// 把 CUDA 返回码转换成 C++ 异常，主函数统一处理错误。
// char const* 是只读 C 字符串，可简单理解为 Java String 的底层形式。
void checkCuda(cudaError_t status, char const* operation)
{
    // cudaSuccess 表示 CUDA 调用成功，其他值都表示失败。
    if (status != cudaSuccess)
    {
        // 拼接“操作名称 + CUDA 错误文本”，再抛给 main 的 catch 处理。
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// CUDA stream 的 RAII 封装，离开作用域时自动释放。
// RAII 可以类比 Java try-with-resources：对象销毁时自动清理底层资源。
class CudaStream
{
public:
    // 构造函数：创建一条 CUDA 命令队列。
    CudaStream()
    {
        // &stream_ 传入字段地址，让 CUDA 把新 stream 写回该字段。
        checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    // 析构函数：对象离开作用域时自动执行，类似 AutoCloseable.close()。
    ~CudaStream()
    {
        // 空指针不需要释放。
        if (stream_ != nullptr)
        {
            // 释放构造函数创建的 CUDA stream。
            cudaStreamDestroy(stream_);
        }
    }

    // 禁止复制 stream 对象，避免两个对象重复释放同一条 stream。
    CudaStream(CudaStream const&) = delete;
    // 同样禁止复制赋值。
    CudaStream& operator=(CudaStream const&) = delete;

    // 允许把 CudaStream 对象直接传给需要 cudaStream_t 的 CUDA/TensorRT 函数。
    operator cudaStream_t() const { return stream_; }

private:
    // 保存 CUDA stream 句柄；nullptr 表示当前没有资源。
    cudaStream_t stream_{nullptr};
};

// GPU 显存的 RAII 封装，避免异常路径泄漏显存。
class DeviceBuffer
{
public:
    // 构造函数接收需要申请的字节数。
    explicit DeviceBuffer(std::size_t bytes)
    {
        // cudaMalloc 在 GPU 上申请显存，并把地址写入 data_。
        checkCuda(cudaMalloc(&data_, bytes), "cudaMalloc");
    }

    // 析构函数负责自动释放 GPU 显存。
    ~DeviceBuffer()
    {
        // 只释放有效地址。
        if (data_ != nullptr)
        {
            // cudaFree 对应前面的 cudaMalloc。
            cudaFree(data_);
        }
    }

    // GPU 显存同样不能被两个对象共同拥有，因此禁止复制。
    DeviceBuffer(DeviceBuffer const&) = delete;
    // 禁止复制赋值，防止重复 cudaFree。
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    // 返回底层 GPU 地址，供 cudaMemcpyAsync 和 TensorRT 使用。
    void* get() const { return data_; }

private:
    // void* 是不指定元素类型的原始地址，因为这里只关心显存起点。
    void* data_{nullptr};
};

// 一次性把 TensorRT engine 读入内存，供 runtime 反序列化。
// const& 表示只读引用：不复制 path，也不允许修改它。
std::vector<char> readFile(std::string const& path)
{
    // binary 表示二进制读取；ate 表示打开后先移动到文件末尾，方便获取大小。
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    // ifstream 可直接作为 bool 判断，false 表示打开失败。
    if (!file)
    {
        // 抛出异常后会跳到 main 最后的 catch。
        throw std::runtime_error("Cannot open engine: " + path);
    }

    // tellg 返回当前位置；因为当前在文件末尾，所以它就是文件字节数。
    auto const end = file.tellg();
    // engine 文件必须至少包含一个字节。
    if (end <= 0)
    {
        throw std::runtime_error("Engine is empty: " + path);
    }

    // 创建一个与文件一样大的连续 char 数组，用于保存全部二进制内容。
    std::vector<char> bytes(static_cast<std::size_t>(end));
    // 把读取位置从文件末尾移回开头。
    file.seekg(0, std::ios::beg);
    // bytes.data() 返回数组首地址；read 把整个文件写入该数组。
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
    {
        throw std::runtime_error("Cannot read engine: " + path);
    }
    // 返回 vector 时编译器会移动数据，通常不会复制整份 engine。
    return bytes;
}

// 返回真正 TensorRT plan 的起始偏移；纯 trtexec engine 的偏移为 0。
std::size_t tensorRtPlanOffset(std::vector<char> const& bytes)
{
    // Ultralytics 文件头格式：int32 JSON 长度 + JSON + TensorRT plan。
    // 文件过短时不可能包含完整元数据头，直接按纯 TensorRT plan 处理。
    if (bytes.size() < sizeof(std::int32_t) + 2)
    {
        return 0;
    }

    // 先准备一个 32 位整数，用来接收 JSON 长度。
    std::int32_t jsonLength = 0;
    // 把文件开头 4 字节复制到 jsonLength；不能直接强转，避免未对齐访问。
    std::memcpy(&jsonLength, bytes.data(), sizeof(jsonLength));
    // JSON 结束位置 = 4 字节长度字段 + JSON 本身长度。
    auto const jsonEnd = sizeof(jsonLength) + static_cast<std::size_t>(std::max(jsonLength, 0));
    // 同时检查长度合法、JSON 以 { 开头并以 } 结尾，降低误判纯 plan 的可能。
    if (jsonLength > 0 && jsonEnd < bytes.size() && bytes[sizeof(jsonLength)] == '{'
        && bytes[jsonEnd - 1] == '}')
    {
        // 返回 JSON 后第一个字节的位置，也就是真正 TensorRT plan 的起点。
        return jsonEnd;
    }
    // 没检测到 Ultralytics 元数据头，plan 从文件第 0 字节开始。
    return 0;
}

// 计算已解析 tensor shape 的元素总数，动态维度没有确定时直接报错。
std::size_t volume(nvinfer1::Dims const& dims)
{
    // nbDims 是维度数量；0 或负数表示 shape 无效。
    if (dims.nbDims <= 0)
    {
        throw std::runtime_error("Tensor has no resolved dimensions");
    }

    // 从 1 开始累乘，例如 [2,300,6] 最终得到 3600。
    std::size_t result = 1;
    // 逐个读取每一个维度。
    for (int i = 0; i < dims.nbDims; ++i)
    {
        // TensorRT 用 -1 表示尚未确定的动态维度，不能据此申请内存。
        if (dims.d[i] <= 0)
        {
            throw std::runtime_error("Tensor still has a dynamic dimension");
        }
        // 把当前维度乘进元素总数。
        result *= static_cast<std::size_t>(dims.d[i]);
    }
    // 返回 tensor 需要容纳的 float 元素数量，不是字节数。
    return result;
}

// 保存 letterbox 参数，用于把网络坐标还原到原图坐标。
// struct 可类比只有字段的 Java DTO。
struct ImageTransform
{
    float scale{};        // 原图缩放比例。
    int left{};           // 左侧填充像素数。
    int top{};            // 顶部填充像素数。
    int originalWidth{};  // 原图宽度。
    int originalHeight{}; // 原图高度。
};

// 第 2~4 步的 OpenCV 预处理函数。
// path 是图片路径，inputHeight/inputWidth 是模型尺寸，chw 指向本张图的输出数组。
ImageTransform preprocess(std::string const& path, int inputHeight, int inputWidth, float* chw)
{
    // 第 2 步：OpenCV 从磁盘读取一张 BGR 原图。
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    // empty() 表示文件不存在、格式不支持或读取失败。
    if (image.empty())
    {
        // 立即停止本次推理，避免把空图片继续传给 resize。
        throw std::runtime_error("Cannot read image: " + path);
    }

    // 第 3 步：保持宽高比缩放，空白区域使用 YOLO 默认的 114 灰色填充。
    // 宽、高两个缩放比例取较小值，确保缩放后的整张图都能放进画布。
    float const scale = std::min(
        // inputWidth / image.cols 是宽度方向允许的最大缩放比例。
        static_cast<float>(inputWidth) / static_cast<float>(image.cols),
        // inputHeight / image.rows 是高度方向允许的最大缩放比例。
        static_cast<float>(inputHeight) / static_cast<float>(image.rows));
    // 原图宽度乘缩放比例并四舍五入，至少保留 1 个像素。
    int const resizedWidth = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    // 原图高度执行相同计算。
    int const resizedHeight = std::max(1, static_cast<int>(std::round(image.rows * scale)));
    // 剩余宽度平均分到左右两侧，这里记录左侧填充。
    int const left = (inputWidth - resizedWidth) / 2;
    // 剩余高度平均分到上下两侧，这里记录顶部填充。
    int const top = (inputHeight - resizedHeight) / 2;

    // 将缩放后的图片居中放到固定输入画布上，这就是 letterbox。
    // resized 先声明为空，cv::resize 会为它分配 CPU 内存。
    cv::Mat resized;
    // 使用双线性插值把原图缩放到刚计算出的尺寸。
    cv::resize(image, resized, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0, cv::INTER_LINEAR);
    // 创建模型大小的三通道画布，并把每个像素初始化为 (114,114,114)。
    cv::Mat canvas(inputHeight, inputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    // Rect 选中画布中央区域，再把 resized 复制进去。
    resized.copyTo(canvas(cv::Rect(left, top, resizedWidth, resizedHeight)));

    // 第 4 步：BGR 转 RGB，再除以 255，得到 0~1 的 float32 数据。
    // rgb 用来保存颜色通道交换后的图片。
    cv::Mat rgb;
    // OpenCV 默认顺序是 BGR，YOLO 训练时使用 RGB，因此交换第 1、3 通道。
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    // uint8 的 0~255 转成 float32 的 0~1，1.0/255.0 是缩放系数。
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    // HWC 转 CHW：cv::split 直接把 R、G、B 三个通道依次写入连续内存。
    // plane 是一个颜色通道包含的元素数，例如 640*640。
    std::size_t const plane = static_cast<std::size_t>(inputHeight) * inputWidth;
    // 三个 Mat 不自己申请内存，而是分别指向 chw 中 R、G、B 的起始位置。
    std::vector<cv::Mat> channels{
        // 第 0 个平面从 chw 开头开始，保存 R。
        cv::Mat(inputHeight, inputWidth, CV_32F, chw),
        // 第 1 个平面偏移 plane 个 float，保存 G。
        cv::Mat(inputHeight, inputWidth, CV_32F, chw + plane),
        // 第 2 个平面偏移 2*plane 个 float，保存 B。
        cv::Mat(inputHeight, inputWidth, CV_32F, chw + 2 * plane)};
    // 把交错排列的 RGBRGB... 拆成连续的 R...G...B...。
    cv::split(rgb, channels);

    // 聚合返回本张图的缩放、填充和原始尺寸，用于第 8 步还原坐标。
    return {scale, left, top, image.cols, image.rows};
}

// 去掉 letterbox 填充和缩放，并把结果裁剪在原图范围内。
float restore(float coordinate, int padding, float scale, int limit)
{
    // 先减去填充，再除以缩放比例；clamp 确保坐标不会落到图片外。
    return std::clamp((coordinate - static_cast<float>(padding)) / scale, 0.0F, static_cast<float>(limit));
}

// 第 8 步：过滤低置信度框、还原原图坐标并打印，不执行 NMS。
// output 是本批输出数组，outputDims 是 shape，activeImages 是真实图片数量。
// imageOffset 是本批在全部路径中的起点，imagePaths/transforms 用于输出图片名和还原坐标。
void printDetections(
    std::vector<float> const& output,
    nvinfer1::Dims const& outputDims,
    std::size_t activeImages,
    std::size_t imageOffset,
    std::vector<std::string> const& imagePaths,
    std::vector<ImageTransform> const& transforms)
{
    // 合法 YOLO26 detect 输出必须是三维 [N,max_det,6]，并覆盖本批真实图片数。
    if (outputDims.nbDims != 3 || outputDims.d[0] < static_cast<std::int64_t>(activeImages)
        || outputDims.d[2] != 6)
    {
        // 输出不符合约定通常表示导出了非 end-to-end 模型或不是 detect 模型。
        throw std::runtime_error("Expected YOLO26 end-to-end output shaped [N, max_det, 6]");
    }

    // 第二维是每张图片的最大检测数量，YOLO26 默认是 300。
    auto const detectionsPerImage = static_cast<std::size_t>(outputDims.d[1]);
    // 外层循环逐张处理当前 batch 中真实存在的图片。
    for (std::size_t batchIndex = 0; batchIndex < activeImages; ++batchIndex)
    {
        // 先打印当前图片路径，\n 表示换行。
        std::cout << "\n" << imagePaths[imageOffset + batchIndex] << '\n';
        // found 用来判断这张图是否至少有一个结果超过阈值。
        bool found = false;
        // 内层循环遍历该图片最多 max_det 个检测结果。
        for (std::size_t detectionIndex = 0; detectionIndex < detectionsPerImage; ++detectionIndex)
        {
            // 每个结果占 6 个 float；指针移动到当前图片、当前检测结果的第一个值。
            float const* detection = output.data() + (batchIndex * detectionsPerImage + detectionIndex) * 6;
            // detection[4] 是置信度，低于阈值就跳到下一条结果。
            if (detection[4] < kConfidenceThreshold)
            {
                continue;
            }

            // 模型输出坐标基于 letterbox 画布，打印前还原到原图。
            // const& 只引用现有 transform，不产生一次结构体复制。
            auto const& transform = transforms[batchIndex];
            // detection[0] 是左上角 x1，去掉左侧 padding 后除以 scale。
            float const x1 = restore(detection[0], transform.left, transform.scale, transform.originalWidth);
            // detection[1] 是左上角 y1，使用顶部 padding 还原。
            float const y1 = restore(detection[1], transform.top, transform.scale, transform.originalHeight);
            // detection[2] 是右下角 x2。
            float const x2 = restore(detection[2], transform.left, transform.scale, transform.originalWidth);
            // detection[3] 是右下角 y2。
            float const y2 = restore(detection[3], transform.top, transform.scale, transform.originalHeight);

            // detection[5] 是类别 ID；置信度打印 3 位小数，坐标打印 1 位小数。
            std::cout << "  class=" << static_cast<int>(detection[5]) << " score=" << std::fixed
                      << std::setprecision(3) << detection[4] << " box=[" << std::setprecision(1) << x1 << ", "
                      << y1 << ", " << x2 << ", " << y2 << "]\n";
            // 记录本张图已经打印过至少一个有效结果。
            found = true;
        }
        // 遍历结束仍未找到结果时，打印明确提示。
        if (!found)
        {
            // 输出当前使用的置信度阈值，便于判断是否需要调低。
            std::cout << "  no detection above " << kConfidenceThreshold << '\n';
        }
    }
}

} // 匿名命名空间

// C++ 程序从 main 开始执行；这里不接收命令行参数，模型和图片路径直接写在代码中。
int main()
{
    /*
     * 从 main 入口开始的完整推理顺序：
     * 第 1 步：在代码中指定模型地址和 N 张原图地址。
     * 第 2 步：OpenCV 读取图片。
     * 第 3 步：letterbox 等比例缩放并填充灰边。
     * 第 4 步：BGR 转 RGB、除以 255、HWC 转 CHW。
     * 第 5 步：把输入从 CPU 复制到 GPU 显存。
     * 第 6 步：TensorRT 执行 YOLO26。
     * 第 7 步：把输出从 GPU 复制回 CPU。
     * 第 8 步：过滤低置信度框、还原原图坐标并打印。
     */

    // C++ 的 try/catch 与 Java 类似；发生异常时还会自动析构前面创建的 RAII 对象。
    try
    {
        // ==================== 第 1 步：直接在这里修改文件地址 ====================

        // TensorRT engine 地址；Windows C++ 字符串建议使用正斜杠，避免写成 \\ 才能表示一个反斜杠。
        std::string const enginePath = "D:/models/yolo26n.engine";

        // N 张原图地址；vector 中写多少个路径，程序就处理多少张图片。
        std::vector<std::string> const imagePaths{
            // 第 1 张图片。
            "D:/images/1.jpg",
            // 第 2 张图片。
            "D:/images/2.jpg",
            // 第 3 张图片；不需要时可以删除这一行，也可以继续往后添加。
            "D:/images/3.jpg",
        };

        // 防止误删所有图片路径后继续执行。
        if (imagePaths.empty())
        {
            // 抛出异常，后面的 catch 会打印错误并返回 1。
            throw std::runtime_error("Please configure at least one image path");
        }

        // ==================== 推理前准备：加载 engine ====================

        // 创建日志器对象，后续 TensorRT 错误会通过它输出。
        Logger logger;
        // readFile 返回 engine 的全部字节；auto 让编译器推断类型为 vector<char>。
        auto engineBytes = readFile(enginePath);
        // 检查并跳过 Ultralytics 可能附加在 plan 前面的 JSON 元数据。
        auto const planOffset = tensorRtPlanOffset(engineBytes);

        // IRuntime 可以类比“模型类加载器”；unique_ptr 表示独占拥有并自动释放它。
        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
        // TensorRT 创建对象失败时返回 nullptr，而不是抛 C++ 异常。
        if (!runtime)
        {
            // 转成异常，交给 main 末尾的 catch 统一处理。
            throw std::runtime_error("createInferRuntime failed");
        }

        // ICudaEngine 是已经针对 GPU 优化、编译完成的模型对象。
        // “runtime->”与 Java 的“runtime.”作用相近，只是 runtime 是智能指针。
        std::unique_ptr<nvinfer1::ICudaEngine> engine{runtime->deserializeCudaEngine(
            // data()+planOffset 指向 plan 起点，size()-planOffset 是 plan 字节数。
            engineBytes.data() + planOffset, engineBytes.size() - planOffset)};
        // 版本不匹配、GPU 不兼容或 engine 损坏都会导致反序列化失败。
        if (!engine)
        {
            throw std::runtime_error("deserializeCudaEngine failed");
        }

        // 通过 TensorRT 11 的 name-based I/O API 自动寻找一个输入和一个输出。
        // 默认构造的 string 是空字符串，稍后写入真实 tensor 名称。
        std::string inputName;
        // YOLO26 detect 端到端模型应当只有一个输出。
        std::string outputName;
        // getNbIOTensors 返回输入和输出 tensor 的总数，循环逐个检查。
        for (int i = 0; i < engine->getNbIOTensors(); ++i)
        {
            // getIOTensorName 返回第 i 个 tensor 的只读 C 字符串名称。
            char const* name = engine->getIOTensorName(i);
            // 判断当前 tensor 是模型输入还是输出。
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            {
                // 已经找到过输入又遇到第二个输入，说明不是本示例支持的单输入模型。
                if (!inputName.empty())
                {
                    throw std::runtime_error("Expected one model input");
                }
                // std::string 会复制 TensorRT 返回的名称文本，后续使用更安全。
                inputName = name;
            }
            else
            {
                // 同理，本示例不处理多个输出 tensor。
                if (!outputName.empty())
                {
                    throw std::runtime_error("Expected one model output");
                }
                // 保存唯一输出的名字，通常是 output0。
                outputName = name;
            }
        }
        // 循环结束后，输入名和输出名都必须已经找到。
        if (inputName.empty() || outputName.empty())
        {
            throw std::runtime_error("Cannot find model input/output");
        }

        // Ultralytics TRT11 的 FP16/INT8 engine 内部是低精度，模型 I/O 仍保持 FP32。
        // c_str() 把 C++ string 临时转换为 TensorRT C API 需要的 char const*。
        if (engine->getTensorDataType(inputName.c_str()) != nvinfer1::DataType::kFLOAT
            || engine->getTensorDataType(outputName.c_str()) != nvinfer1::DataType::kFLOAT)
        {
            // 本示例 host buffer 使用 vector<float>，因此只接受 float32 I/O。
            throw std::runtime_error("This minimal example expects FP32 model I/O");
        }

        // kDEVICE 表示这两个 tensor 必须绑定 GPU 地址，而不是普通 CPU 地址。
        if (engine->getTensorLocation(inputName.c_str()) != nvinfer1::TensorLocation::kDEVICE
            || engine->getTensorLocation(outputName.c_str()) != nvinfer1::TensorLocation::kDEVICE)
        {
            throw std::runtime_error("This minimal example expects device I/O tensors");
        }

        // 示例只支持 YOLO26 detect 的 NCHW 图片输入。
        // Dims 中 nbDims 是维度数量，d[0..3] 分别是 N、C、H、W。
        nvinfer1::Dims const modelInputDims = engine->getTensorShape(inputName.c_str());
        // NCHW 必须正好有 4 个维度。
        if (modelInputDims.nbDims != 4)
        {
            throw std::runtime_error("Expected NCHW input");
        }

        // 动态 engine 从 profile 0 读取 min/opt/max；图片尺寸使用 opt，batch 使用 max 分批。
        // 先假设所有维度都是固定值。
        bool dynamicInput = false;
        // 逐维查找 -1；TensorRT 使用 -1 表示运行时才能确定的动态维度。
        for (int i = 0; i < modelInputDims.nbDims; ++i)
        {
            // 只要任意一维小于 0，dynamicInput 就会变成 true。
            dynamicInput = dynamicInput || modelInputDims.d[i] < 0;
        }

        // 三元表达式“条件 ? A : B”相当于 Java 的同名三元运算符。
        // 动态模型读取 profile 的最小 shape；静态模型直接使用固定 shape。
        nvinfer1::Dims const minInputDims = dynamicInput
            ? engine->getProfileShape(inputName.c_str(), 0, nvinfer1::OptProfileSelector::kMIN)
            : modelInputDims;
        // opt shape 是构建 engine 时重点优化的尺寸，本示例用它的 H 和 W。
        nvinfer1::Dims const optInputDims = dynamicInput
            ? engine->getProfileShape(inputName.c_str(), 0, nvinfer1::OptProfileSelector::kOPT)
            : modelInputDims;
        // max shape 给出该 engine 允许的最大 batch 和最大图片尺寸。
        nvinfer1::Dims const maxInputDims = dynamicInput
            ? engine->getProfileShape(inputName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX)
            : modelInputDims;
        // NCHW 的 d[1] 是通道数，普通 RGB YOLO 模型必须是 3。
        if (optInputDims.d[1] != 3)
        {
            throw std::runtime_error("Expected a 3-channel NCHW input");
        }

        // 静态 batch 的最后一批会复制最后一张图补齐；只打印真实图片的结果。
        // d[0] 为 -1 表示 N 可以在 profile 范围内变化。
        bool const dynamicBatch = modelInputDims.d[0] < 0;
        // 动态模型读取最小 N；静态模型最小 N 就是固定 N。
        auto const minBatch = static_cast<std::size_t>(dynamicBatch ? minInputDims.d[0] : modelInputDims.d[0]);
        // 动态模型读取最大 N，稍后用它切分任意数量的图片。
        auto const maxBatch = static_cast<std::size_t>(dynamicBatch ? maxInputDims.d[0] : modelInputDims.d[0]);
        // d[2] 是输入高度；static_cast<int> 是显式类型转换。
        int const inputHeight = static_cast<int>(optInputDims.d[2]);
        // d[3] 是输入宽度。
        int const inputWidth = static_cast<int>(optInputDims.d[3]);
        // 拒绝 batch 为 0、范围颠倒或图片尺寸无效的 profile。
        if (minBatch == 0 || maxBatch < minBatch || inputHeight <= 0 || inputWidth <= 0)
        {
            throw std::runtime_error("Invalid optimization profile");
        }

        // context 持有执行状态；同一条 stream 串联 profile、拷贝和 enqueueV3。
        // IExecutionContext 可以类比 Java 推理框架中的 Session，一次次执行同一个 engine。
        std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
        // context 创建失败一般表示显存不足或 engine 与当前设备不兼容。
        if (!context)
        {
            throw std::runtime_error("createExecutionContext failed");
        }

        // 构造 CudaStream，同时在构造函数中创建底层 CUDA stream。
        CudaStream stream;
        // 选择第 0 个 optimization profile，并让后续 shape 设置使用这个 profile。
        if (!context->setOptimizationProfileAsync(0, stream))
        {
            throw std::runtime_error("setOptimizationProfileAsync failed");
        }

        // 打印程序从 engine 中读取到的输入名、最优尺寸、最大 batch 和输出名。
        std::cout << "input=" << inputName << " shape=[N,3," << inputHeight << ',' << inputWidth
                  << "] max_batch=" << maxBatch << " output=" << outputName << '\n';

        // 输入图片多于 engine 最大 batch 时，循环拆成多个 batch。
        // 一张图片包含 3*H*W 个 float；3ULL 让乘法使用足够大的无符号整数类型。
        std::size_t const imageElements = 3ULL * inputHeight * inputWidth;
        // imageOffset 是当前 batch 第一张图在 imagePaths 中的位置。
        for (std::size_t imageOffset = 0; imageOffset < imagePaths.size();)
        {
            // activeImages 是本批真实图片数，不能超过 engine 的 maxBatch。
            std::size_t const activeImages = std::min(maxBatch, imagePaths.size() - imageOffset);
            // executionBatch 是实际送入模型的 N；必要时会大于 activeImages，用复制图片补齐。
            std::size_t const executionBatch = dynamicBatch ? std::max(activeImages, minBatch) : maxBatch;

            // 每批先设置实际 N，H/W 使用导出时的最优尺寸。
            // 复制 opt shape，避免直接修改只读的 optInputDims。
            nvinfer1::Dims inputDims = optInputDims;
            // 把 N 改成本批真正执行的 batch size。
            inputDims.d[0] = static_cast<std::int64_t>(executionBatch);
            // 把本批实际输入 shape 写入 TensorRT context。
            if (!context->setInputShape(inputName.c_str(), inputDims))
            {
                throw std::runtime_error("setInputShape failed");
            }

            // 第 2~4 步：逐张调用 OpenCV 完成读取、letterbox 和数据格式转换。
            // input 是 CPU 上的连续 float 数组，布局为 [N][C][H][W]。
            std::vector<float> input(executionBatch * imageElements);
            // transforms 与图片一一对应，保存各自的 letterbox 参数。
            std::vector<ImageTransform> transforms;
            // reserve 只预留容量，不增加元素数量，减少 push_back 时的重新分配。
            transforms.reserve(executionBatch);
            // 只对本批真实图片执行磁盘读取和 OpenCV 预处理。
            for (std::size_t i = 0; i < activeImages; ++i)
            {
                // input.data()+i*imageElements 定位第 i 张图片在大数组中的起点。
                // preprocess 返回本张图的坐标还原参数，push_back 把它加入 vector。
                transforms.push_back(preprocess(
                    imagePaths[imageOffset + i], inputHeight, inputWidth, input.data() + i * imageElements));
            }

            // 静态 batch 或 profile 最小 batch 大于实际图片数时，用最后一张图补齐。
            // 补齐的图片只用于满足模型 shape，不会在第 8 步打印结果。
            for (std::size_t i = activeImages; i < executionBatch; ++i)
            {
                // copy_n 把最后一张真实图片的 CHW float 数据复制到第 i 个位置。
                std::copy_n(
                    input.data() + (activeImages - 1) * imageElements, imageElements, input.data() + i * imageElements);
                // 补齐图片复用最后一张真实图片的 transform。
                transforms.push_back(transforms.back());
            }

            // 输入 shape 确定后，TensorRT 才能给出本批次的实际输出 shape。
            // YOLO26 默认得到 [executionBatch,300,6]。
            nvinfer1::Dims const outputDims = context->getTensorShape(outputName.c_str());
            // 在 CPU 上创建足够大的 float 输出数组，初始值为 0。
            std::vector<float> output(volume(outputDims));
            // 元素数乘 sizeof(float) 转成字节数，再申请输入 GPU 显存。
            DeviceBuffer inputDevice(input.size() * sizeof(float));
            // 为输出申请另一块 GPU 显存；两块显存在本轮循环结束时自动释放。
            DeviceBuffer outputDevice(output.size() * sizeof(float));
            // 把 TensorRT 输入名绑定到输入显存地址，输出名绑定到输出显存地址。
            // setTensorAddress 只登记地址，本身不复制任何数据。
            if (!context->setTensorAddress(inputName.c_str(), inputDevice.get())
                || !context->setTensorAddress(outputName.c_str(), outputDevice.get()))
            {
                throw std::runtime_error("setTensorAddress failed");
            }

            // 第 5 步：把预处理后的输入从 CPU 内存复制到 GPU 显存（H2D）。
            // cudaMemcpyAsync 只是把复制任务加入 stream，不会在这一行等待复制完成。
            checkCuda(cudaMemcpyAsync(inputDevice.get(), input.data(), input.size() * sizeof(float),
                          // 明确复制方向是 Host(CPU) -> Device(GPU)，并指定同一条 stream。
                          cudaMemcpyHostToDevice, stream),
                // 发生 CUDA 错误时，该文本会出现在异常消息中。
                "copy input to GPU");

            // 第 6 步：TensorRT 在 GPU 上执行 YOLO26 网络。
            // enqueueV3 同样只是把推理任务排到 stream；同一 stream 保证它排在 H2D 后面。
            if (!context->enqueueV3(stream))
            {
                throw std::runtime_error("enqueueV3 failed");
            }

            // 第 7 步：把 TensorRT 输出从 GPU 显存复制回 CPU 内存（D2H）。
            // 该复制任务排在推理后面，因此会等 GPU 推理完成后再读取输出显存。
            checkCuda(cudaMemcpyAsync(output.data(), outputDevice.get(), output.size() * sizeof(float),
                          // 复制方向是 Device(GPU) -> Host(CPU)。
                          cudaMemcpyDeviceToHost, stream),
                "copy output to CPU");
            // CPU 在这里阻塞，直到 stream 中的 H2D、推理和 D2H 三个任务全部完成。
            checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

            // 第 8 步：过滤、坐标还原和打印；YOLO26 端到端输出不需要 NMS。
            // 只传 activeImages，因此补齐图片的输出不会被打印。
            printDetections(output, outputDims, activeImages, imageOffset, imagePaths, transforms);
            // 移动到下一批真实图片；例如本批处理 8 张，就把 offset 增加 8。
            imageOffset += activeImages;
        }
    }
    // const& 避免复制异常对象；std::exception 是大多数标准异常的父类。
    catch (std::exception const& error)
    {
        // what() 返回异常消息，写到标准错误输出。
        std::cerr << "Error: " << error.what() << '\n';
        // 返回非 0 表示程序执行失败。
        return 1;
    }
    // 所有 batch 都成功处理后返回 0，表示程序正常结束。
    return 0;
}
