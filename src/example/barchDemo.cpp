#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <opencv2/opencv.hpp>
#include "infer.h"
#include "cpm.h"
#include "yolo.h"
#include "config.h"

using namespace std;

static yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

shared_ptr<yolo::Infer> myYolo;
cudaStream_t customStream;

bool initBatch(const string &engine_file, float * &confidences, const float &nms) {
    myYolo = yolo::load(engine_file, confidences, nms, customStream);
    if (myYolo == nullptr) {
        cout << "================================= TensorRT INIT FAIL =================================" << endl;
        return false;
    } else {
        // 创建非阻塞流
        cudaStreamCreate(&customStream);
        cout << "================================= TensorRT INIT SUCCESS =================================" << endl;
        return true;
    }
}

vector<vector<yolo::Box> > inferBatch(const vector<cv::Mat> &mats) {
    trt::Timer timer;
    vector<yolo::Image> yoloimages(mats.size());
    transform(mats.begin(), mats.end(), yoloimages.begin(), cvimg);
    timer.start(customStream);
    auto objs = myYolo->forwards(yoloimages, customStream);
    vector<vector<yolo::Box> > res;
    for (int i = 0; i < objs.size(); ++i) {
        res.emplace_back(objs[i]);
    }
    timer.stop("batch 12");
    return res;
}

int main() {
    auto *confidence_thresholds = new float[80];
    for (int i = 0; i < 80; i++) {
        confidence_thresholds[i] = 0.25;
    }

    Config config;
    int batch = 12;
    initBatch(config.MODEL, confidence_thresholds, 0.7);

    vector<cv::Mat> images;
    auto img = cv::imread(config.TEST_IMG);
    for (int i = 0; i < batch; i++) {
        images.push_back(img);
    }

    while (true) {
        vector<vector<yolo::Box> > batched_result = inferBatch(images);
        // for (int ib = 0; ib < (int) batched_result.size(); ++ib) {
        //     auto &objs = batched_result[ib];
        //     for (auto &obj: objs) {
        //         auto name = obj.class_label;
        //         auto caption = cv::format("%i %.2f", name, obj.confidence);
        //         cout << "class_label: " << name << " confidence: " << caption << " (L T R B): (" << obj.left << ", "
        //                 << obj.top << ", " << obj.right << ", " << obj.bottom << ")" << endl;
        //     }
        //     printf("all: %d objects\n", (int) objs.size());
        // }
    }

    return 0;
}
