#include <opencv2/opencv.hpp>
#include "infer.hpp"
#include "cpm.hpp"
#include "yolo.hpp"
#include "config.cpp"

using namespace std;

static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

static yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

bool initBatchAsync(const string &engine_file, const float &confidence, const float &nms, const int &max_batch) {
    bool ok = cpmi.start([&engine_file, &confidence, &nms] {
        return yolo::load(engine_file, yolo::Type::V8, confidence, nms);
    }, max_batch);
    if (!ok) {
        cout << "================================= TensorRT INIT FAIL =================================" << endl;
        return false;
    } else {
        cout << "================================= TensorRT INIT SUCCESS =================================" << endl;
        return true;
    }
}

vector<vector<yolo::Box>> inferBatchAsync(const vector<cv::Mat> &mats) {
    trt::Timer timer;
    vector<yolo::Image> yoloimages(mats.size());
    transform(mats.begin(), mats.end(), yoloimages.begin(), cvimg);
    timer.start();
    auto objs = cpmi.commits(yoloimages);
    vector<vector<yolo::Box>> res;
    for (int i = 0; i < objs.size(); ++i) {
        res.emplace_back(objs[i].get());
    }
    timer.stop("batch 12");
    return res;
}

//int main() {
//    Config config;
//    initBatchAsync(config.MODEL, 0.25, 0.7, 6);
//
//    vector<cv::Mat> images{
//            cv::imread(config.TEST_IMG),
//            cv::imread(config.TEST_IMG),
//            cv::imread(config.TEST_IMG),
//            cv::imread(config.TEST_IMG),
//            cv::imread(config.TEST_IMG),
//            cv::imread(config.TEST_IMG)
//    };
//
//
//    while (true) {
//        vector<vector<yolo::Box>> batched_result = inferBatchAsync(images);
////        for (int ib = 0; ib < (int) batched_result.size(); ++ib) {
////            auto &objs = batched_result[ib];
////            for (auto &obj: objs) {
////                auto name = config.cocolabels[obj.class_label];
////                auto caption = cv::format("%s %.2f", name, obj.confidence);
////                cout << "class_label: " << name << " confidence: " << caption << " (L T R B): (" << obj.left << ", "
////                     << obj.top << ", " << obj.right << ", " << obj.bottom << ")" << endl;
////            }
////            printf("all: %d objects\n", (int) objs.size());
////        }
//    }
//
//    return 0;
//
//}

