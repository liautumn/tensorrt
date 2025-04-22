#include "export_common.h"

bool initSingle(const string &engineFile, const float confidence, const float nms, const int gpu_device) {
    // 创建非阻塞流
    cudaStreamCreate(&cudaStream);
    my_yolo = yolo::load(engineFile, confidence, nms, gpu_device, cudaStream);
    if (my_yolo != nullptr) {
        //预热
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
    // 调用推理函数获取检测框
    std::vector<detect::Box> boxes = inferSingle(mat);
    // 设置返回的大小
    *size = boxes.size();
    // 为结果分配内存
    *result = new detect::Box[boxes.size()];
    // 拷贝结果到分配的内存中
    std::copy(boxes.begin(), boxes.end(), *result);
}

EXPORT_API void TENSORRT_SINGLE_DESTROY() {
    my_yolo.reset();
}
