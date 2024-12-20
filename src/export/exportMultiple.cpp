#include <opencv2/opencv.hpp>
#include "infer.h"
#include "cpm.h"
#include "yolo.h"

using namespace std;

//static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

static yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

static shared_ptr<yolo::Infer> my_yolo;

extern "C" __declspec(dllexport) bool TENSORRT_Multiple_INIT(const char *engineFile, float confidence, float nms, int maxBatch) {
//    bool ok = cpmi.start([&engine_file, &confidence, &nms] {
//        return yolo::load(engine_file, yolo::Type::V8, confidence, nms);
//    }, maxBatch);
    my_yolo = yolo::load(engineFile, yolo::Type::V8, confidence, nms);
    if (my_yolo == nullptr) {
        return false;
    } else {
        return true;
    }
}

extern "C" __declspec(dllexport) void TENSORRT_Multiple_INFER(cv::Mat **mats, int imgSize, yolo::Box ***result, int **resultSizes) {
    trt::Timer timer;
    timer.start();
    vector<yolo::Image> yoloimages;
    for (int i = 0; i < imgSize; ++i) {
        yoloimages.emplace_back(mats[i]->data, mats[i]->cols, mats[i]->rows);
    }
//    auto batched_result = cpmi.commits(yoloimages);
    auto batched_result = my_yolo->forwards(yoloimages);
    timer.stop("batch n");

    *resultSizes = new int[batched_result.size()];
    *result = new yolo::Box *[batched_result.size()];

    for (int ib = 0; ib < (int) batched_result.size(); ++ib) {
        auto &boxes = batched_result[ib];
        (*resultSizes)[ib] = boxes.size();
        (*result)[ib] = new yolo::Box[boxes.size()];
        memcpy((*result)[ib], boxes.data(), boxes.size() * sizeof(yolo::Box));
    }
}