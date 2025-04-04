#include <opencv2/opencv.hpp>
#include <filesystem>
#include "infer.h"
#include "yolo.h"
#include "config.h"

using namespace std;
namespace fs = std::filesystem;

void syncInfer() {
    auto *confidence_thresholds = new float[84];
    for (int i = 0; i < 84; i++) {
        confidence_thresholds[i] = 0.25;
    }

    Config config;
    auto yolo = yolo::load(config.MODEL, confidence_thresholds, 0.4);
    if (yolo == nullptr) return;

    //预热
    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->forward(yrImage);
    }

    trt::Timer timer;

    // 创建一个窗口
    // std::string windowName = "Image Window";
    // cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    // int width_ = 1024; // 设置窗口宽度
    // int height = 640; // 设置窗口高度
    // cv::resizeWindow(windowName, width_, height);

    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);
    while (true) {
        timer.start();
        auto objs = yolo->forward(image);
        timer.stop("batch one");
        // for (auto &obj: objs) {
        //     cout << "class_label: " << obj.class_label << " caption: " << obj.confidence << " (L T R B): (" << obj.left
        //             << ", "
        //             << obj.top << ", " << obj.right << ", " << obj.bottom << ")" << endl;
        //
        //     uint8_t b, g, r;
        //
        //     tie(b, g, r) = yolo::random_color(obj.class_label);
        //     rectangle(mat, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
        //               cv::Scalar(b, g, r), 5);
        //
        //     auto name = obj.class_label;
        //     auto caption = cv::format("%i %.2f", name, obj.confidence);
        //     int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        //     rectangle(mat, cv::Point(obj.left - 3, obj.top - 33),
        //               cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
        //     putText(mat, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
        //             16);
        // }
        // cv::imshow(windowName, mat); // 显示帧
        // cv::waitKey(0);
    }
}

int main() {
    syncInfer();
    return 0;
}
