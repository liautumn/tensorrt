#include "export_common.h"

bool initSingleCpm(const string &engineFile, const float confidence, const float nms, const int gpu_device) {
    // ������������
    cudaStreamCreate(&cudaStream);
    bool ok = cpmi.start([&engineFile, &confidence, &nms, &gpu_device] {
        return yolo::load(engineFile, confidence, nms, gpu_device, cudaStream);
    }, 1, cudaStream);
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

EXPORT_API bool TENSORRT_SINGLE_CPM_INIT(const char *engineFile, float confidence, float nms, const int gpu_device) {
    return initSingleCpm(engineFile, confidence, nms, gpu_device);
}

EXPORT_API void TENSORRT_SINGLE_CPM_INFER(cv::Mat *mat, detect::Box **result, int *size) {
    // ������������ȡ����
    std::vector<detect::Box> boxes = inferSingleCpm(mat);
    // ���÷��صĴ�С
    *size = boxes.size();
    // Ϊ��������ڴ�
    *result = new detect::Box[boxes.size()];
    // ���������������ڴ���
    std::copy(boxes.begin(), boxes.end(), *result);
}

EXPORT_API void TENSORRT_SINGLE_CPM_DESTROY() {
    cpmi.stop();
}
