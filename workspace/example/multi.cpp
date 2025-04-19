#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <timer.h>
#include "infer.h"
#include "cpm.h"
#include "yolo.h"
#include "config.h"

using namespace std;

static yolo::Image cvimg(const cv::Mat &image) { return {image.data, image.cols, image.rows}; }

shared_ptr<yolo::Infer> my_yolo;
cudaStream_t cudaStream2;

bool initBatch(const string &engine_file, const float &confidence, const float &nms) {
    Config config;
    cudaStreamCreate(&cudaStream2);
    my_yolo = yolo::load(engine_file, confidence, nms, config.GPU_DEVICE, cudaStream2);
    if (my_yolo == nullptr) {
        cout << "================================= TensorRT INIT FAIL =================================" << endl;
        return false;
    } else {
        cout << "================================= TensorRT INIT SUCCESS =================================" << endl;
        return true;
    }
}

vector<vector<detect::Box> > inferBatch(const vector<cv::Mat> &mats) {
    trt_timer::Timer timer;
    vector<yolo::Image> yolo_images(mats.size());
    transform(mats.begin(), mats.end(), yolo_images.begin(), cvimg);
    timer.start(cudaStream2);
    auto objs = my_yolo->detect_forwards(yolo_images, cudaStream2);
    vector<vector<detect::Box> > res;
    for (int i = 0; i < objs.size(); ++i) {
        res.emplace_back(objs[i]);
    }
    timer.stop("batch 12");
    return res;
}

// int main()
// {
//     Config config;
//     int batch = 12;
//     initBatch(config.MODEL, 0.2, 0.5);
//
//     vector<cv::Mat> images;
//     auto img = cv::imread(config.TEST_IMG);
//     for (int i = 0; i < batch; i++)
//     {
//         images.push_back(img);
//     }
//
//     // while (true)
//     // {
//         vector<vector<detect::Box>> batched_result = inferBatch(images);
//         for (int ib = 0; ib < (int) batched_result.size(); ++ib) {
//             auto &objs = batched_result[ib];
//             for (auto &obj: objs) {
//                 auto name = obj.class_label;
//                 auto caption = cv::format("%i %.2f", name, obj.confidence);
//                 cout << "class_label: " << name << " confidence: " << caption << " (L T R B): (" << obj.left << ", "
//                         << obj.top << ", " << obj.right << ", " << obj.bottom << ")" << endl;
//             }
//             printf("all: %d objects\n", (int) objs.size());
//         }
//     // }
//
//     return 0;
// }
