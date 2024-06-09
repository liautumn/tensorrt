#include <opencv2/opencv.hpp>
#include "infer.hpp"
#include "cpm.hpp"
#include "yolo.hpp"

using namespace std;

static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

bool initAsync2(const string &engine_file, float confidence, float nms) {
    bool ok = cpmi.start([&engine_file, &confidence, &nms] {
        return yolo::load(engine_file, yolo::Type::V8, confidence, nms);
    });
    if (!ok) {
        cout << "================================= TensorRT INIT FAIL =================================" << endl;
        return false;
    } else {
        cout << "================================= TensorRT INIT SUCCESS =================================" << endl;
        return true;
    }
}

vector<yolo::Box> inferAsync2(cv::Mat *mat) {
    trt::Timer timer;
    auto img = yolo::Image(mat->data, mat->cols, mat->rows);
    timer.start();
    auto objs = cpmi.commit(img).get();
    timer.stop("batch one");
    return objs;
}

extern "C" __declspec(dllexport) bool TensorRT_INIT_ASYNC2(const char *engine_file, float confidence, float nms) {
    return initAsync2(engine_file, confidence, nms);
}
extern "C" __declspec(dllexport) void TensorRT_INFER_ASYNC2(cv::Mat *mat, yolo::Box **result, int *size) {
    // ������������ȡ����
    std::vector<yolo::Box> boxes = inferAsync2(mat);
    // ���÷��صĴ�С
    *size = boxes.size();
    // Ϊ��������ڴ�
    *result = new yolo::Box[boxes.size()];
    // ���������������ڴ���
    std::copy(boxes.begin(), boxes.end(), *result);
}