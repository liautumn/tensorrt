#include <cuda_runtime.h>
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
cudaStream_t customStream1;

void syncInfer() {
    auto *confidence_thresholds = new float[80];
    for (int i = 0; i < 80; i++) {
        confidence_thresholds[i] = 0.25;
    }

    cudaStream_t customStream;
    // 创建非阻塞流
    cudaStreamCreate(&customStream);

    Config config;
    auto yolo = yolo::load(config.MODEL, confidence_thresholds, 0.4, customStream);
    if (yolo == nullptr) return;

    //预热
    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->forward(yrImage, customStream);
    }

    trt_timer::Timer Timer;

    // 创建一个窗口
    // std::string windowName = "Image Window";
    // cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    // int width_ = 1024; // 设置窗口宽度
    // int height = 640; // 设置窗口高度
    // cv::resizeWindow(windowName, width_, height);

    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);
    while (true) {
        Timer.start(customStream);
        auto objs = yolo->forward(image, customStream);
        Timer.stop("batch one");
    }
    // for (auto &obj: objs) {
    //     cout << "class_label: " << obj.class_label << " caption: " << obj.confidence << " (L T R B): (" << obj.left
    //             << ", "
    //             << obj.top << ", " << obj.right << ", " << obj.bottom << ")" << endl;
    //
    //     rectangle(mat, cv::Point(static_cast<int>(obj.left), static_cast<int>(obj.top)),
    //               cv::Point(static_cast<int>(obj.right), static_cast<int>(obj.bottom)),
    //               cv::Scalar(255, 0, 255), 5);
    //
    //     auto name = obj.class_label;
    //     auto caption = cv::format("%i %.2f", name, obj.confidence);
    //     int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    //     rectangle(mat, cv::Point(static_cast<int>(obj.left) - 3, static_cast<int>(obj.top) - 33),
    //               cv::Point(static_cast<int>(obj.left) + width, static_cast<int>(obj.top)), cv::Scalar(255, 0, 255),
    //               -1);
    //     putText(mat, caption, cv::Point(static_cast<int>(obj.left), static_cast<int>(obj.top) - 5), 0, 1,
    //             cv::Scalar::all(0), 2,
    //             16);
    // }
    // cv::imshow(windowName, mat); // 显示帧
    // cv::waitKey(0);
}

bool initSingleCpm(const string &engineFile, float *confidences, float nms) {
    // 创建非阻塞流
    cudaStreamCreate(&customStream1);
    bool ok = cpmi.start([&engineFile, &confidences, &nms] {
        return yolo::load(engineFile, confidences, nms, customStream1);
    }, 1, customStream1);
    if (!ok) {
        return false;
    } else {
        //预热
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
        for (int i = 0; i < 10; ++i) {
            cpmi.commit(yrImage).get();
        }
        return true;
    }
}

vector<yolo::Box> inferSingleCpm(const cv::Mat &mat) {
    return cpmi.commit(yolo::Image(mat.data, mat.cols, mat.rows)).get();
}

void asyncInfer() {
    auto *confidence_thresholds = new float[80];
    for (int i = 0; i < 80; i++) {
        confidence_thresholds[i] = 0.25;
    }
    Config config;
    if (initSingleCpm(config.MODEL, confidence_thresholds, 0.5)) {
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
    syncInfer();
    // asyncInfer();
    return 0;
}
