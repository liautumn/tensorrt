// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

#include "batch.h"
#include "image.h"
#include "inference.h"
#include "model.h"
#include "preprocess.h"
#include "result.h"
#include "timer.h"
#include "validator.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {
    // 下面的辅助函数只在本文件可见；Model 的所有权仍由 main() 持有。

    // 一次完整调用的 CPU 结果和三段耗时。对象只在线程自己的 runDetection 调用中存活。
    struct DetectionRun {
        Results results; // 当前帧或当前批次的结构化检测结果。
        double preprocessMilliseconds{}; // CPU 计时覆盖的预处理阶段耗时。
        float inferenceMilliseconds{}; // CUDA event 测得的 GPU 推理耗时。
        double postprocessMilliseconds{}; // CPU 后处理和坐标还原耗时。
    };

    // 图片目录和视频帧共同使用的预处理、推理及后处理流程。
    [[nodiscard]] DetectionRun runDetection(
        Model &model, // 当前 worker 独占使用的 Model 实例。
        std::span<cv::Mat const> const images, // 所有输入图片的只读视图。
        Batch const &batch, // 当前批次在 images 中的 offset 和 size。
        float confidenceThreshold, // 过滤低置信度检测框的阈值。
        trt_timer::Timer &inferenceTimer) { // 当前 worker 独有的 CUDA event 计时器。
        // 一个 Model 的 context、workspace 和 stream 必须由同一条 worker 线程按顺序使用。
        // 这一步只修改当前实例的动态输入 shape，不会触碰其他 Model。
        setBatchSize(model, batch.size);

        // 使用 steady_clock 计量包含同步点的 CPU 预处理总耗时。
        auto const preprocessStart = std::chrono::steady_clock::now();
        // 预处理把输入写入当前 Model 的 inputDevice，并返回本批次的逆仿射矩阵。
        AffineMatrices affineMatrices = preprocessBatchToGpu(model, images, batch);
        // preprocessBatchToGpu 返回前已同步该实例的 stream，矩阵此时可供 CPU 后处理使用。
        auto const preprocessEnd = std::chrono::steady_clock::now();

        // 计时事件绑定到当前实例的 stream；不同 Model 的事件不会互相配对。
        inferenceTimer.start(model.stream.get());
        // enqueueV3 在同一 stream 上提交 TensorRT 工作，infer() 返回前等待它完成。
        infer(model);
        // 只复制当前 batch 的有效输出，避免把 maxBatch 的尾部无效区域带回 CPU。
        std::vector<float> output = copyToCpu(model, batch.size);
        // stop() 会同步结束事件，因此此处得到的是本次推理阶段的 GPU 时间。
        float const inferenceMilliseconds = inferenceTimer.stop("inference", false);

        // 后处理只读取 CPU output 和 affineMatrices，不再访问 TensorRT context。
        auto const postprocessStart = std::chrono::steady_clock::now();
        Results results = printBatchResults(
            model,
            batch,
            affineMatrices,
            output,
            confidenceThreshold);
        // 记录坐标还原、阈值过滤和结果结构化的结束时间。
        auto const postprocessEnd = std::chrono::steady_clock::now();

        // 移动结果集合，避免从局部对象复制检测框。
        return {
            .results = std::move(results), // 将后处理结果交给调用方。
            .preprocessMilliseconds
            = std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count(),
            .inferenceMilliseconds = inferenceMilliseconds, // CUDA event 的推理耗时。
            .postprocessMilliseconds
            = std::chrono::duration<double, std::milli>(postprocessEnd - postprocessStart).count()
        };
    }

    void printTiming(
        std::string_view const itemName, // 当前统计对象，例如 batch 或 frame。
        std::size_t itemIndex, // 对象在对应序列中的下标。
        int batchSize, // 这次调用实际送入 TensorRT 的图片数量。
        DetectionRun const &run) { // 需要打印的三段耗时。
        // 先在本地流中完成格式化，避免直接修改进程级 cout 的 flags/precision。
        std::ostringstream message;
        message << std::fixed << std::setprecision(3)
                << "timing: " << itemName << '=' << itemIndex
                << " batch=" << batchSize
                << " preprocess=" << run.preprocessMilliseconds << " ms"
                << " inference=" << run.inferenceMilliseconds << " ms"
                << " postprocess=" << run.postprocessMilliseconds << " ms"
                << " total="
                << run.preprocessMilliseconds
                + run.inferenceMilliseconds
                + run.postprocessMilliseconds
                << " ms";
        // Validator 会添加当前 model 标签；日志接口不加锁，极高并发下行间可能交错。
        Validator::info(message.str());
    }

    // 只允许 OpenCV 运行时支持的常见图片扩展名进入目录模式。
    [[nodiscard]] bool isImagePath(std::filesystem::path const &path) {
        // 扩展名比较统一使用小写，避免 Windows/Linux 文件名大小写差异影响筛选。
        static constexpr std::array<std::string_view, 7> supportedExtensions{
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
        };

        // path.extension() 只提取最后一个后缀，不读取文件内容。
        std::string extension = path.extension().string();
        // 原地把后缀转换成小写，随后与静态扩展名表比较。
        std::ranges::transform(
            extension,
            extension.begin(),
            [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

        // 返回该后缀是否属于支持的图片类型。
        return std::ranges::find(supportedExtensions, extension) != supportedExtensions.end();
    }

    // 枚举并排序目录中的图片，保证不同批次和结果下标保持稳定。
    [[nodiscard]] ImagePaths findImagePaths(std::filesystem::path const &imageDirectory) {
        // 错误信息使用字符串副本，便于传给 printf 风格的 Assertf。
        std::string const directoryText = imageDirectory.string();
        // 目录不存在时立即失败，不让 directory_iterator 抛出缺少上下文的异常。
        Assertf(std::filesystem::is_directory(imageDirectory),
                "Image directory does not exist: %s", directoryText.c_str());

        // 只保存路径，不在枚举阶段读取图片像素。
        ImagePaths imagePaths;
        // 遍历目录中的每个目录项，后续只保留普通文件和支持的扩展名。
        for (std::filesystem::directory_entry const &entry
             : std::filesystem::directory_iterator(imageDirectory)) {
            if (entry.is_regular_file() && isImagePath(entry.path())) {
                // 保留原始 path 对象，loadImages 会按此顺序读取。
                imagePaths.push_back(entry.path());
            }
        }

        // 排序让目录遍历顺序不依赖文件系统实现。
        std::ranges::sort(imagePaths);
        // 空目录没有可推理输入，直接报告目录路径。
        Assertf(!imagePaths.empty(),
                "No supported images found in directory: %s", directoryText.c_str());
        // 返回稳定排序后的图片路径集合。
        return imagePaths;
    }

    void detectImageDirectory(
        Model &model, // 目录模式使用的单个 Model；调用方不能同时复用它。
        std::filesystem::path const &imageDirectory, // 待读取图片所在目录。
        float confidenceThreshold) { // 后处理使用的置信度阈值。
        // 线程局部日志标签和 CUDA 当前 device 都在进入业务流程时重新绑定。
        LogContext logContext(model.name);
        selectCudaDevice(model.deviceId);
        // 先收集路径，再一次性读取图片，保证结果下标与排序后的路径一致。
        ImagePaths const imagePaths = findImagePaths(imageDirectory);
        // images 持有原始 BGR 像素，后续批次只借用其只读视图。
        Images images = loadImages(imagePaths);
        // 根据 Engine 的 maxBatch 把全部图片划分成连续批次。
        Batches const batches = splitByMaxBatch(images.size(), model.maxBatch);

        // 全局结果按图片顺序累积，最终与 images 一一对应。
        Results results;
        // 预留图片数量，降低跨批次 push_back 的扩容次数。
        results.reserve(images.size());
        // 计时器只服务于当前线程和当前 Model 的 stream。
        trt_timer::Timer inferenceTimer;

        // 目录模式按批次串行执行，不在同一 Model 上建立后台流水线。
        for (std::size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
            // 只借用 batches 中的当前元素，不复制批次描述。
            Batch const &batch = batches[batchIndex];
            // 完成该批次的设置、预处理、推理和后处理。
            DetectionRun run = runDetection(
                model,
                images,
                batch,
                confidenceThreshold,
                inferenceTimer);
            // 输出当前批次耗时；目录模式下这里只有一个调用线程。
            printTiming("batch", batchIndex, batch.size, run);
            // 把当前批次的外层结果移动到全局集合，保持图片原始顺序。
            std::ranges::move(run.results, std::back_inserter(results));
        }

        // 所有批次完成后统一创建结果窗口并显示，不再访问 TensorRT 输出显存。
        // 窗口名称带上 modelA/modelB，避免不同模型的结果下标相互冲突。
        showResults(images, results, model.name);
    }

    void detectVideo(
        Model &model, // 当前 worker 独占的模型运行态。
        std::filesystem::path const &videoPath, // 两个 worker 可各自打开同一路径。
        float confidenceThreshold) { // 当前 worker 的后处理阈值。
        // 新线程不会继承 CUDA 当前 device，必须在创建 Timer 和 VideoCapture 前显式绑定。
        LogContext logContext(model.name);
        selectCudaDevice(model.deviceId);
        // VideoCapture、帧缓存和 Timer 都是本 worker 的局部对象，不在两个模型之间共享。
        // 将 path 转成 OpenCV 需要的字符串；不会修改 main() 中的原始路径。
        std::string const videoPathText = videoPath.string();
        // 每个 worker 独立维护一个解码器，因此当前示例会把同一视频读取两次。
        cv::VideoCapture capture(videoPathText);
        // 解码器初始化失败时终止当前线程函数；异常未捕获会导致进程终止。
        Assertf(capture.isOpened(), "Cannot open video: %s", videoPathText.c_str());

        // 直接使用 Model 保存的实例名，避免调用处重复维护 modelA/modelB 字符串。
        std::string const &windowName = model.name;
        // 窗口名必须唯一；HighGUI 后端仍是进程级共享设施，见 README 的线程安全说明。
        // 生产环境应考虑把这些 GUI 调用集中到主线程。
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);

        // 视频逐帧推理固定使用 batch=1，offset 始终指向本轮唯一的 frame。
        constexpr Batch batch{.offset = 0, .size = 1};
        // CUDA event 属于当前 worker，并绑定 runDetection 使用的 model.stream。
        trt_timer::Timer inferenceTimer;
        // frameIndex 用于日志下标和 FPS 平滑公式。
        std::size_t frameIndex = 0;
        // 指数平滑后的 FPS，首帧之前没有历史值。
        double smoothedFps = 0.0;

        // capture.read 每次把下一帧写入局部 frame；循环结束表示视频已读完。
        for (cv::Mat frame; capture.read(frame); ++frameIndex) {
            // 从读取帧开始计量完整的“推理 + 绘制 + 显示”周期。
            auto const frameStart = std::chrono::steady_clock::now();
            // Images 只保存当前帧的 cv::Mat 头，像素仍由 frame 持有。
            Images images{frame};
            // 使用当前 Model 的独立 context、显存和 stream 推理这一帧。
            DetectionRun run = runDetection(
                model,
                images,
                batch,
                confidenceThreshold,
                inferenceTimer);
            // timing 输出可能与另一个 worker 的 stdout 输出交错。
            printTiming("frame", frameIndex, batch.size, run);

            // 在原始帧副本上绘制检测框，避免修改 capture 返回的 frame。
            cv::Mat displayFrame = drawImageResults(frame, run.results.front());
            // 记录绘制前后的时间，得到用户实际看到的帧周期。
            auto const frameEnd = std::chrono::steady_clock::now();
            // 把时间间隔转换为秒，防止以毫秒计算时单位错误。
            double const frameSeconds
                    = std::chrono::duration<double>(frameEnd - frameStart).count();
            // 避免极端情况下除以零；正常帧的 FPS 是周期倒数。
            double const currentFps = frameSeconds > 0.0 ? 1.0 / frameSeconds : 0.0;
            // 首帧直接使用当前值，后续使用 90% 历史值和 10% 当前值。
            smoothedFps = frameIndex == 0
                              ? currentFps
                              : smoothedFps * 0.9 + currentFps * 0.1;

            // 组装要覆盖在视频帧左上角的 FPS 标签。
            std::ostringstream fpsLabel;
            fpsLabel << "FPS: " << std::fixed << std::setprecision(1) << smoothedFps;
            // 在显示副本上绘制绿色 FPS 文本，不影响检测结果结构体。
            cv::putText(
                displayFrame,
                fpsLabel.str(),
                cv::Point(16, 32),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 0),
                2,
                cv::LINE_AA);
            // 提交当前 worker 的窗口刷新；HighGUI 后端在进程内仍然是共享的。
            cv::imshow(windowName, displayFrame);

            // waitKey(1) 同时处理窗口事件并给出约 1 ms 的刷新机会。
            int const key = cv::waitKey(1);
            // ESC/q/Q 只结束当前 worker 的视频循环，另一个 worker 不会自动停止。
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        // 关闭当前 worker 创建的窗口；不会替另一个 worker 关闭其窗口。
        cv::destroyWindow(windowName);
    }
} // namespace

int main() {
    // Windows 示例使用 win.engine；Linux 运行时应改成 model/linux.engine。
    std::filesystem::path const enginePath
            = R"(D:\autumn\Documents\CLionProjects\tensorrt\model\win.engine)";
    // 两个 worker 都打开这份视频，各自维护独立的解码状态。
    std::filesystem::path const videoPath
            = R"(D:\autumn\Documents\CLionProjects\tensorrt\model\test.mp4)";
    // 目录模式备用路径；默认视频模式下暂未使用。
    std::filesystem::path const imageDirectory
            = R"(D:\autumn\Documents\CLionProjects\tensorrt\model)";
    // 两个模型使用相同的后处理置信度阈值；lambda 按值捕获该常量。
    constexpr float confidenceThreshold = 0.25F;
    // true 运行双模型视频 worker；false 依次运行两个模型的图片目录模式。
    constexpr bool runVideo = true;
    // 两个值相同表示在同一张 GPU 上用不同 stream 并发；有第二张 GPU 时可把 modelBDevice 改为 1。
    constexpr int modelADevice = 0;
    constexpr int modelBDevice = 0;

    // EngineData 只读共享：两个 initModel 调用各自反序列化，不共享 runtime/engine/context。
    // 读取只发生在主线程，worker 启动后不再访问这段 plan 字节。
    EngineData engineData = readEngine(enginePath);

    // 每个 Model 独占一套 TensorRT 对象、CUDA stream、I/O 显存和预处理 workspace。
    // 这里两个实例加载的是同一个 plan，并不是两个不同网络结构的模型。
    Model modelA;
    // initModel 会在 modelA 内创建 runtime、engine、context、stream 和各类缓冲区。
    initModel(modelA, engineData, "modelA", modelADevice);
    Model modelB;
    // 第二次调用重新反序列化同一份 plan，所有运行态地址与 modelA 分离。
    initModel(modelB, engineData, "modelB", modelBDevice);

    if (runVideo)
    {
        // 一条 jthread 只持有一个 Model 的引用，因此两个实例可在各自 stream 上并行调度。
        // jthread 的析构会先等待线程结束；它们声明在 Model 之后，保证 Model 不会提前销毁。
        // lambda 中只捕获引用和阈值，不复制不可移动的 Model。
        std::jthread t1([&modelA, &videoPath, confidenceThreshold] {
            // t1 只调用 modelA；detectVideo 内部会再次绑定 modelADevice。
            detectVideo(modelA, videoPath, confidenceThreshold);
        });
        std::jthread t2([&modelB, &videoPath, confidenceThreshold] {
            // t2 只调用 modelB；它可以与 t1 通过独立 stream 并行竞争 GPU 资源。
            detectVideo(modelB, videoPath, confidenceThreshold);
        });
    }
    else
    {
        // 目录模式明确调用两个模型；顺序显示可避免两个 worker 同时驱动 HighGUI 窗口。
        detectImageDirectory(modelA, imageDirectory, confidenceThreshold);
        detectImageDirectory(modelB, imageDirectory, confidenceThreshold);
    }

    // 离开作用域时先析构 t2/t1（request_stop + join），再析构 modelB/modelA。
    // 因而正常退出路径不会在线程仍使用 Model 时释放其 CUDA/TensorRT 资源。
    return 0;
}
