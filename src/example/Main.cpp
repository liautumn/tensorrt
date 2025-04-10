#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "Yolo.h"
#include "Config.h"
#include "Cpm.h"
#include "Timer.h"

using namespace std;
namespace fs = std::filesystem;

static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
cudaStream_t cudaStream1;

void syncInfer() {
    cudaStreamCreate(&cudaStream1);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.2, 0.4, cudaStream1);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->forward(yrImage, cudaStream1);
    }

    std::string windowName = "Image Window";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    int width_ = 1024;
    int height = 640;
    cv::resizeWindow(windowName, width_, height);

    trt_timer::Timer Timer;
    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);
    Timer.start(cudaStream1);
    auto objs = yolo->forward(image, cudaStream1);
    Timer.stop("batch one");
    for (auto &obj: objs) {
        cout << "class_label: " << obj.class_label << " caption: " << obj.confidence << " (L T R B): (" << obj.left
                << ", "
                << obj.top << ", " << obj.right << ", " << obj.bottom << ")" << endl;

        rectangle(mat, cv::Point(static_cast<int>(obj.left), static_cast<int>(obj.top)),
                  cv::Point(static_cast<int>(obj.right), static_cast<int>(obj.bottom)),
                  cv::Scalar(255, 0, 255), 2);

        auto name = obj.class_label;
        auto caption = cv::format("%i %.2f", name, obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 1, nullptr).width + 10;
        rectangle(mat, cv::Point(static_cast<int>(obj.left) - 3, static_cast<int>(obj.top) - 33),
                  cv::Point(static_cast<int>(obj.left) + width, static_cast<int>(obj.top)), cv::Scalar(255, 0, 255),
                  -1);
        putText(mat, caption, cv::Point(static_cast<int>(obj.left), static_cast<int>(obj.top) - 5), 0, 1,
                cv::Scalar::all(0), 1,
                16);
    }
    cv::imshow(windowName, mat);
    cv::waitKey(0);
}

// 检测结果绘制函数
void draw_detection_results(cv::Mat &mat,
                            const yolo::BoxArray &objs) {
    for (const auto &obj: objs) {
        // 过滤低置信度的检测结果
        if (obj.confidence < 0.5) continue;

        // 绘制边界框
        cv::rectangle(mat,
                      cv::Point(static_cast<int>(obj.left), static_cast<int>(obj.top)),
                      cv::Point(static_cast<int>(obj.right), static_cast<int>(obj.bottom)),
                      cv::Scalar(255, 0, 255), // 品红色边框
                      2); // 线宽

        // 构造标注文本
        std::string label = cv::format("%i: %.1f%%",
                                       obj.class_label,
                                       obj.confidence * 100);

        // 计算文本尺寸
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label,
                                             cv::FONT_HERSHEY_SIMPLEX,
                                             0.5,
                                             1,
                                             &baseline);

        // 绘制文本背景
        cv::rectangle(mat,
                      cv::Point(static_cast<int>(obj.left),
                                static_cast<int>(obj.top) - text_size.height - 10),
                      cv::Point(static_cast<int>(obj.left) + text_size.width,
                                static_cast<int>(obj.top)),
                      cv::Scalar(255, 0, 255),
                      cv::FILLED);

        // 绘制文本
        cv::putText(mat,
                    label,
                    cv::Point(static_cast<int>(obj.left),
                              static_cast<int>(obj.top) - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar::all(0),
                    1,
                    16);
    }
}

void videoDemo() {
    cudaStreamCreate(&cudaStream1);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.1, 0.4, cudaStream1);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->forward(yrImage, cudaStream1);
    }

    // 视频文件路径配置
    const std::string video_path = "D:/autumn/Downloads/001.mp4"; // 确保路径正确

    // 打开视频流（优先尝试作为文件打开）
    cv::VideoCapture cap(video_path);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1); // 减少缓冲延迟

    // 获取视频属性
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Video Info: " << std::endl;
    std::cout << " - Resolution: " << frame_width << "x" << frame_height << std::endl;
    std::cout << " - Original FPS: " << fps << std::endl;

    cv::Mat mat;
    yolo::Image image;
    trt_timer::Timer Timer;

    // 创建显示窗口
    cv::namedWindow("YOLO Video Detection",
                    cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow("YOLO Video Detection", 1280, 720);

    // 时间统计变量
    double total_processed_time = 0.0;
    int processed_frames = 0;

    double avg_fps;

    while (true) {
        // 精确计时开始
        double start_time = cv::getTickCount();

        // 读取视频帧
        cap >> mat;
        if (mat.empty()) break;

        // 转换为YOLO处理格式
        image = yolo::Image(mat.data, mat.cols, mat.rows);

        // CUDA加速推理
        Timer.start(cudaStream1);
        auto objs = yolo->forward(image, cudaStream1);
        Timer.stop("batch one");

        // 绘制检测结果
        draw_detection_results(mat, objs);

        // 计算处理耗时
        double process_time = (cv::getTickCount() - start_time) / cv::getTickFrequency();
        total_processed_time += process_time;
        processed_frames++;

        // 计算并显示实时FPS
        double current_fps = 1.0 / process_time;
        avg_fps = processed_frames / total_processed_time;

        // 在帧上叠加信息
        cv::putText(mat,
                    cv::format("FPS: %.1f | Avg: %.1f", current_fps, avg_fps),
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 255, 0),
                    2);

        // 显示结果
        cv::imshow("YOLO Video Detection", mat);

        // 处理退出按键
        const int key = cv::waitKey(1);
        if (key == 27) {
            // ESC键退出
            break;
        } else if (key == '+' && processed_frames > 0) {
            // 动态调整显示FPS
            std::cout << "Average FPS reset" << std::endl;
            total_processed_time = 0.0;
            processed_frames = 0;
        }
    }

    // 最终统计信息
    std::cout << "Processing completed!" << std::endl;
    std::cout << "Total frames: " << processed_frames << std::endl;
    std::cout << "Average FPS: " << avg_fps << std::endl;

    // 释放资源
    cap.release();
    cv::destroyAllWindows();
}

bool initSingleCpm(const string &engineFile, float confidence, float nms) {
    cudaStreamCreate(&cudaStream1);
    bool ok = cpmi.start([&engineFile, &confidence, &nms] {
        return yolo::load(engineFile, confidence, nms, cudaStream1);
    }, 1, cudaStream1);
    if (ok) {
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
        for (int i = 0; i < 10; ++i) {
            cpmi.commit(yrImage).get();
        }
        return true;
    } else {
        return false;
    }
}

vector<yolo::Box> inferSingleCpm(const cv::Mat &mat) {
    return cpmi.commit(yolo::Image(mat.data, mat.cols, mat.rows)).get();
}

void asyncInfer() {
    Config config;
    if (initSingleCpm(config.MODEL, 0.2, 0.5)) {
        trt_timer::Timer Timer;
        const cv::Mat mat = cv::imread(config.TEST_IMG);
        while (true) {
            Timer.start();
            inferSingleCpm(mat);
            Timer.stop("batch one");
        }
    }
}

int main() {
    // syncInfer();
    // asyncInfer();
    videoDemo();
    return 0;
}
