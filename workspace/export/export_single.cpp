#include "export_common.h"

bool initSingle(const string &engineFile, const float confidence, const float nms, const int gpu_device) {
    // ������������
    cudaStreamCreate(&cudaStream);
    my_yolo = yolo::load(engineFile, confidence, nms, gpu_device, cudaStream);
    if (my_yolo != nullptr) {
        //Ԥ��
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
        for (int i = 0; i < 10; ++i) {
            my_yolo->detect_forward(yrImage, cudaStream);
        }
        return true;
    } else {
        return false;
    }
}

vector<detect::Box> inferSingle(cv::Mat *mat) {
    trt_timer::Timer timer;
    timer.start(cudaStream);
    auto img = yolo::Image(mat->data, mat->cols, mat->rows);
    auto objs = my_yolo->detect_forward(img, cudaStream);
    timer.stop("batch one");
    return objs;
}

EXPORT_API bool TENSORRT_SINGLE_INIT(const char *engineFile, float confidence, float nms, int gpuDevice) {
    return initSingle(engineFile, confidence, nms, gpuDevice);
}

EXPORT_API void TENSORRT_SINGLE_INFER(cv::Mat *mat, detect::Box **result, int *size) {
    // ������������ȡ����
    std::vector<detect::Box> boxes = inferSingle(mat);
    // ���÷��صĴ�С
    *size = boxes.size();
    // Ϊ��������ڴ�
    *result = new detect::Box[boxes.size()];
    // ���������������ڴ���
    std::copy(boxes.begin(), boxes.end(), *result);
}

EXPORT_API void TENSORRT_SINGLE_DESTROY() {
    my_yolo.reset();
}
