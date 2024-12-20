#include <opencv2/opencv.hpp>
#include "infer.h"
#include "cpm.h"
#include "yolo.h"

using namespace std;

static shared_ptr<yolo::Infer> yolo1;

bool initSingle(const string &engineFile, float confidence, float nms) {
    yolo1 = yolo::load(engineFile, yolo::Type::V8, confidence, nms);
    if (yolo1 != nullptr) {
        //Ԥ��
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
        for (int i = 0; i < 10; ++i) {
            yolo1->forward(yrImage);
        }
        return true;
    } else {
        return false;
    }
}

vector<yolo::Box> inferSingle(cv::Mat *mat) {
    trt::Timer timer;
    timer.start();
    auto img = yolo::Image(mat->data, mat->cols, mat->rows);
    auto objs = yolo1->forward(img);
    timer.stop("batch one");
    return objs;
}

extern "C" __declspec(dllexport) bool TENSORRT_SINGLE_INIT(const char *engineFile, float confidence, float nms) {
    return initSingle(engineFile, confidence, nms);
}
extern "C" __declspec(dllexport) void TENSORRT_SINGLE_INFER(cv::Mat *mat, yolo::Box **result, int *size) {
    // ������������ȡ����
    std::vector<yolo::Box> boxes = inferSingle(mat);
    // ���÷��صĴ�С
    *size = boxes.size();
    // Ϊ��������ڴ�
    *result = new yolo::Box[boxes.size()];
    // ���������������ڴ���
    std::copy(boxes.begin(), boxes.end(), *result);
}
extern "C" __declspec(dllexport) void TENSORRT_SINGLE_DESTROY() {
    yolo1.reset();
}