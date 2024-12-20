#include <opencv2/opencv.hpp>
#include "infer.h"
#include "cpm.h"
#include "yolo.h"

using namespace std;

static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

bool initSingleCpm(const string &engineFile, float confidence, float nms) {
    bool ok = cpmi.start([&engineFile, &confidence, &nms] {
        return yolo::load(engineFile, yolo::Type::V8, confidence, nms);
    });
    if (!ok) {
        return false;
    } else {
        //Ԥ��
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
        for (int i = 0; i < 10; ++i) {
            cpmi.commit(yrImage).get();
        }
        return true;
    }
}

vector<yolo::Box> inferSingleCpm(cv::Mat *mat) {
    return cpmi.commit(yolo::Image(mat->data, mat->cols, mat->rows)).get();
}

extern "C" __declspec(dllexport) bool TENSORRT_SINGLE_CPM_INIT(const char *engineFile, float confidence, float nms) {
    return initSingleCpm(engineFile, confidence, nms);
}
extern "C" __declspec(dllexport) void TENSORRT_SINGLE_CPM_INFER(cv::Mat *mat, yolo::Box **result, int *size) {
    // ������������ȡ����
    std::vector<yolo::Box> boxes = inferSingleCpm(mat);
    // ���÷��صĴ�С
    *size = boxes.size();
    // Ϊ��������ڴ�
    *result = new yolo::Box[boxes.size()];
    // ���������������ڴ���
    std::copy(boxes.begin(), boxes.end(), *result);
}
extern "C" __declspec(dllexport) void TENSORRT_SINGLE_CPM_DESTROY() {
    cpmi.stop();
}