#include "export_common.h"

EXPORT_API bool TENSORRT_MULTIPLE_INIT(const char *engineFile, const float confidence, const float nms, int maxBatch,
                                       const int gpuDevice) {
    // 创建非阻塞流
    cudaStreamCreate(&cudaStream);
    my_yolo = yolo::load(engineFile, confidence, nms, gpuDevice, cudaStream);
    if (my_yolo != nullptr) {
        //预热
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        vector<yolo::Image> inputs;
        inputs.reserve(maxBatch);
        for (int i = 0; i < maxBatch; ++i) {
            inputs.emplace_back(yrMat.data, yrMat.cols, yrMat.rows);
        }
        for (int i = 0; i < 10; ++i) {
            auto batched_result = my_yolo->detect_forwards(inputs, cudaStream);
        }
        return true;
    }
    return false;
}

EXPORT_API void TENSORRT_MULTIPLE_INFER(cv::Mat **mats, int imgSize, detect::Box ***result, int **resultSizes) {
    trt_timer::Timer timer;
    timer.start(cudaStream);
    vector<yolo::Image> inputs;
    inputs.reserve(imgSize);
    for (int i = 0; i < imgSize; ++i) {
        inputs.emplace_back(mats[i]->data, mats[i]->cols, mats[i]->rows);
    }
    auto batched_result = my_yolo->detect_forwards(inputs, cudaStream);
    timer.stop("batch n");

    *resultSizes = new int[batched_result.size()];
    *result = new detect::Box *[batched_result.size()];

    for (int ib = 0; ib < static_cast<int>(batched_result.size()); ++ib) {
        auto &boxes = batched_result[ib];
        (*resultSizes)[ib] = boxes.size();
        (*result)[ib] = new detect::Box[boxes.size()];
        memcpy((*result)[ib], boxes.data(), boxes.size() * sizeof(detect::Box));
    }
}

EXPORT_API void TENSORRT_MULTIPLE_DESTROY() {
    my_yolo.reset();
}
