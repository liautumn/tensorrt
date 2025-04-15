#include <cuda_runtime.h>
#include <driver_types.h>
#include <opencv2/opencv.hpp>
#include <timer.h>
#include "infer.h"
#include "cpm.h"
#include "yolo.h"

using namespace std;

static shared_ptr<yolo::Infer> yolo2;
cudaStream_t customStream3;

extern "C" __declspec(dllexport) bool
TENSORRT_MULTIPLE_INIT(const char *engineFile, const float confidence, const float nms, int maxBatch) {
    // 创建非阻塞流
    cudaStreamCreate(&customStream3);
    yolo2 = yolo::load(engineFile, confidence, nms, customStream3);
    if (yolo2 != nullptr) {
        //预热
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        vector<yolo::Image> inputs;
        inputs.reserve(maxBatch);
        for (int i = 0; i < maxBatch; ++i) {
            inputs.emplace_back(yrMat.data, yrMat.cols, yrMat.rows);
        }
        for (int i = 0; i < 5; ++i) {
            auto batched_result = yolo2->forwards(inputs, customStream3);
        }
        return true;
    }
    return false;
}

extern "C" __declspec(dllexport) void
TENSORRT_MULTIPLE_INFER(cv::Mat **mats, int imgSize, detect::Box ***result, int **resultSizes) {
    trt_timer::Timer timer;
    timer.start();
    vector<yolo::Image> inputs;
    inputs.reserve(imgSize);
    for (int i = 0; i < imgSize; ++i) {
        inputs.emplace_back(mats[i]->data, mats[i]->cols, mats[i]->rows);
    }
    auto batched_result = yolo2->forwards(inputs, customStream3);
    timer.stop("batch n");

    *resultSizes = new int[batched_result.size()];
    *result = new detect::Box *[batched_result.size()];

    for (int ib = 0; ib < (int) batched_result.size(); ++ib) {
        auto &boxes = batched_result[ib];
        (*resultSizes)[ib] = boxes.size();
        (*result)[ib] = new detect::Box[boxes.size()];
        memcpy((*result)[ib], boxes.data(), boxes.size() * sizeof(detect::Box));
    }
}
extern "C" __declspec(dllexport) void TENSORRT_MULTIPLE_DESTROY() {
    yolo2.reset();
}