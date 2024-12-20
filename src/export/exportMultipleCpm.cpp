#include <opencv2/opencv.hpp>
#include "infer.h"
#include "cpm.h"
#include "yolo.h"

using namespace std;

static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi2;

extern "C" __declspec(dllexport) bool
TENSORRT_MULTIPLE_CPM_INIT(const char *engineFile, float confidence, float nms, int maxBatch) {
    bool ok = cpmi2.start([&engineFile, &confidence, &nms] {
        return yolo::load(engineFile, yolo::Type::V8, confidence, nms);
    }, maxBatch);
    if (ok) {
        //‘§»»
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        vector<yolo::Image> inputs;
        inputs.reserve(maxBatch);
        for (int i = 0; i < maxBatch; ++i) {
            inputs.emplace_back(yrMat.data, yrMat.cols, yrMat.rows);
        }
        for (int i = 0; i < 5; ++i) {
            auto list = cpmi2.commits(inputs);
            for (const auto &item: list) {
                item.get();
            }
        }
    }
    return ok;
}

extern "C" __declspec(dllexport) void
TENSORRT_MULTIPLE_CPM_INFER(cv::Mat **mats, int imgSize, yolo::Box ***result, int **resultSizes) {
    vector<yolo::Image> inputs;
    inputs.reserve(imgSize);
    for (int i = 0; i < imgSize; ++i) {
        inputs.emplace_back(mats[i]->data, mats[i]->cols, mats[i]->rows);
    }
    auto batched_result = cpmi2.commits(inputs);

    *resultSizes = new int[batched_result.size()];
    *result = new yolo::Box *[batched_result.size()];

    for (int ib = 0; ib < (int) batched_result.size(); ++ib) {
        auto &boxes = batched_result[ib].get();
        (*resultSizes)[ib] = boxes.size();
        (*result)[ib] = new yolo::Box[boxes.size()];
        memcpy((*result)[ib], boxes.data(), boxes.size() * sizeof(yolo::Box));
    }
}
extern "C" __declspec(dllexport) void TENSORRT_MULTIPLE_CPM_DESTROY() {
    cpmi2.stop();
}