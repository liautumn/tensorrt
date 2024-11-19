#include <opencv2/opencv.hpp>
#include "infer.h"
#include "cpm.h"
#include "yolo.h"

using namespace std;

static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

bool initAsyncNew(const string &engineFile, float confidence, float nms) {
    bool ok = cpmi.start([&engineFile, &confidence, &nms] {
        return yolo::load(engineFile, yolo::Type::V8, confidence, nms);
    });
    if (!ok) {
        INFO("================================= TensorRT INIT FAIL =================================");
        return false;
    } else {
        INFO("================================= TensorRT INIT SUCCESS =================================");
        //Ԥ��
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
        for (int i = 0; i < 10; ++i) {
            cpmi.commit(yrImage).get();
        }
        INFO("10 times of warm-up completed");
        return true;
    }
}

vector<yolo::Box> inferAsyncNew(cv::Mat *mat) {
//    trt::Timer timer;
    auto img = yolo::Image(mat->data, mat->cols, mat->rows);
//    timer.start();
    auto objs = cpmi.commit(img).get();
//    timer.stop("batch one");
    return objs;
}

extern "C" __declspec(dllexport) bool TENSORRT_INIT_ASYNC_NEW(const char *engineFile, float confidence, float nms) {
    return initAsyncNew(engineFile, confidence, nms);
}
extern "C" __declspec(dllexport) void TENSORRT_INFER_ASYNC_NEW(cv::Mat *mat, yolo::Box **result, int *size) {
    // ������������ȡ����
    std::vector<yolo::Box> boxes = inferAsyncNew(mat);
    // ���÷��صĴ�С
    *size = boxes.size();
    // Ϊ��������ڴ�
    *result = new yolo::Box[boxes.size()];
    // ���������������ڴ���
    std::copy(boxes.begin(), boxes.end(), *result);
}
extern "C" __declspec(dllexport) void TENSORRT_STOP_NEW() {
    cpmi.stop();
}