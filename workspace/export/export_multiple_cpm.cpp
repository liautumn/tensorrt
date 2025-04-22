#include "export_common.h"

EXPORT_API bool TENSORRT_MULTIPLE_CPM_INIT(const char *engineFile, const float confidence, const float nms,
                                           int maxBatch, const int gpuDevice) {
    // 创建非阻塞流
    cudaStreamCreate(&cudaStream);
    bool ok = cpmi.start([&engineFile, &confidence, &nms, &gpuDevice] {
        return yolo::load(engineFile, confidence, nms, gpuDevice, cudaStream);
    }, maxBatch, cudaStream);
    if (ok) {
        //预热
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        vector<yolo::Image> inputs;
        inputs.reserve(maxBatch);
        for (int i = 0; i < maxBatch; ++i) {
            inputs.emplace_back(yrMat.data, yrMat.cols, yrMat.rows);
        }
        for (int i = 0; i < 10; ++i) {
            auto list = cpmi.commits(inputs);
            for (const auto &item: list) {
                item.get();
            }
        }
    }
    return ok;
}

EXPORT_API void TENSORRT_MULTIPLE_CPM_INFER(cv::Mat **mats, int imgSize, detect::Box ***result, int **resultSizes) {
    vector<yolo::Image> inputs;
    inputs.reserve(imgSize);
    for (int i = 0; i < imgSize; ++i) {
        inputs.emplace_back(mats[i]->data, mats[i]->cols, mats[i]->rows);
    }
    auto batched_result = cpmi.commits(inputs);

    *resultSizes = new int[batched_result.size()];
    *result = new detect::Box *[batched_result.size()];

    for (int ib = 0; ib < static_cast<int>(batched_result.size()); ++ib) {
        auto &boxes = batched_result[ib].get();
        (*resultSizes)[ib] = boxes.size();
        (*result)[ib] = new detect::Box[boxes.size()];
        memcpy((*result)[ib], boxes.data(), boxes.size() * sizeof(detect::Box));
    }
}

EXPORT_API void TENSORRT_MULTIPLE_CPM_DESTROY() {
    cpmi.stop();
}
