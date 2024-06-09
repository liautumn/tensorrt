#include <opencv2/opencv.hpp>
#include "infer.hpp"
#include "cpm.hpp"
#include "yolo.hpp"

using namespace std;

//static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

static yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

static shared_ptr<yolo::Infer> my_yolo;

extern "C" __declspec(dllexport) bool
initBatchAsync(const char *engine_file, float confidence, float nms, int max_batch) {
//    bool ok = cpmi.start([&engine_file, &confidence, &nms] {
//        return yolo::load(engine_file, yolo::Type::V8, confidence, nms);
//    }, max_batch);
//    if (!ok) {
    my_yolo = yolo::load(engine_file, yolo::Type::V8, confidence, nms);
    if (my_yolo == nullptr) {
        cout << "================================= TensorRT INIT FAIL =================================" << endl;
        return false;
    } else {
        cout << "================================= TensorRT INIT SUCCESS =================================" << endl;
        return true;
    }
}

extern "C" __declspec(dllexport) void
inferBatchAsync(cv::Mat **mats, int imgSize, yolo::Box ***result, int **resultSizes) {
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
