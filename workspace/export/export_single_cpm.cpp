#include <cuda_runtime.h>
#include <driver_types.h>
#include <opencv2/opencv.hpp>
#include "infer.h"
#include "cpm.h"
#include "yolo.h"

using namespace std;

static cpm::Instance<detect::BoxArray, yolo::Image, yolo::Infer> cpmi;
cudaStream_t customStream2;

bool initSingleCpm(const int gpu_device, const string &engineFile, const float confidence, const float nms) {
    // ������������
    cudaStreamCreate(&customStream2);
    bool ok = cpmi.start([&gpu_device, &engineFile, &confidence, &nms] {
        return yolo::load(gpu_device, engineFile, confidence, nms, customStream2);
    }, 1, customStream2);
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

vector<detect::Box> inferSingleCpm(cv::Mat *mat) {
    return cpmi.commit(yolo::Image(mat->data, mat->cols, mat->rows)).get();
}

extern "C" __declspec(dllexport) bool TENSORRT_SINGLE_CPM_INIT(const int gpu_device, const char *engineFile,
                                                               float confidence, float nms) {
    return initSingleCpm(gpu_device, engineFile, confidence, nms);
}

extern "C" __declspec(dllexport) void TENSORRT_SINGLE_CPM_INFER(cv::Mat *mat, detect::Box **result, int *size) {
    // ������������ȡ����
    std::vector<detect::Box> boxes = inferSingleCpm(mat);
    // ���÷��صĴ�С
    *size = boxes.size();
    // Ϊ��������ڴ�
    *result = new detect::Box[boxes.size()];
    // ���������������ڴ���
    std::copy(boxes.begin(), boxes.end(), *result);
}

extern "C" __declspec(dllexport) void TENSORRT_SINGLE_CPM_DESTROY() {
    cpmi.stop();
}
