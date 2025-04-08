#include <cuda_runtime.h>
#include <driver_types.h>
#include <opencv2/opencv.hpp>
#include "infer.h"
#include "cpm.h"
#include "yolo.h"
#include <Timer.h>

using namespace std;

static shared_ptr<yolo::Infer> yolo1;
cudaStream_t customStream1;

bool initSingle(const string &engineFile, const float confidence, const float nms) {
    // 创建非阻塞流
    cudaStreamCreate(&customStream1);
    yolo1 = yolo::load(engineFile, confidence, nms, customStream1);
    if (yolo1 != nullptr) {
        //预热
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
        for (int i = 0; i < 10; ++i) {
            yolo1->forward(yrImage, customStream1);
        }
        return true;
    } else {
        return false;
    }
}

vector<yolo::Box> inferSingle(cv::Mat *mat) {
    trt_timer::Timer timer;
    timer.start();
    auto img = yolo::Image(mat->data, mat->cols, mat->rows);
    auto objs = yolo1->forward(img, customStream1);
    timer.stop("batch one");
    return objs;
}

extern "C" __declspec(dllexport) bool TENSORRT_SINGLE_INIT(const char *engineFile, float *confidences, float nms) {
    return initSingle(engineFile, confidences, nms);
}

extern "C" __declspec(dllexport) void TENSORRT_SINGLE_INFER(cv::Mat *mat, yolo::Box **result, int *size) {
    // 调用推理函数获取检测框
    std::vector<yolo::Box> boxes = inferSingle(mat);
    // 设置返回的大小
    *size = boxes.size();
    // 为结果分配内存
    *result = new yolo::Box[boxes.size()];
    // 拷贝结果到分配的内存中
    std::copy(boxes.begin(), boxes.end(), *result);
}

extern "C" __declspec(dllexport) void TENSORRT_SINGLE_DESTROY() {
    yolo1.reset();
}
