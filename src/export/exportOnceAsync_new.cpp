#include <opencv2/opencv.hpp>
#include "infer.hpp"
#include "cpm.hpp"
#include "yolo.hpp"

using namespace std;

static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

bool initAsyncNew(const string &engineFile, float confidence, float nms) {
    bool ok = cpmi.start([&engineFile, &confidence, &nms] {
        return yolo::load(engineFile, yolo::Type::V8, confidence, nms);
    });
    if (!ok) {
        cout << "================================= TensorRT INIT FAIL =================================" << endl;
        return false;
    } else {
        cout << "================================= TensorRT INIT SUCCESS =================================" << endl;
        return true;
    }
}

vector<yolo::Box> inferAsyncNew(cv::Mat *mat) {
    trt::Timer timer;
    auto img = yolo::Image(mat->data, mat->cols, mat->rows);
    timer.start();
    auto objs = cpmi.commit(img).get();
    timer.stop("batch one");
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