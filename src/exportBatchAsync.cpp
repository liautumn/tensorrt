#include <opencv2/opencv.hpp>
#include "infer.hpp"
#include "cpm.hpp"
#include "yolo.hpp"

using namespace std;

static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

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

vector<yolo::BoxArray> inferBatchAsync(vector<uchar> &images) {
    trt::Timer timer;
    vector<yolo::Image> yoloimages(images.size());
    std::for_each(images.begin(), images.end(), [&](const auto &item) {
        yoloimages.push_back(yolo::Image(&item, 1920, 1200));
    });
    timer.start();
    auto objs = cpmi.commits(yoloimages);
    vector<yolo::BoxArray> res(objs.size());
    for (int i = 0; i < objs.size(); ++i) {
        res.push_back(objs[i].get());
    }
    timer.stop("batch 12");
    return res;
}

// int main() {
//     Config config;
//     initBatchAsync(config.MODEL, 0.45, 0.5, 3);

//    vector<cv::Mat> images{
//            cv::imread(config.TEST_IMG),
//            cv::imread(config.TEST_IMG),
//            cv::imread(config.TEST_IMG)
//    };
//
//    vector<yolo::BoxArray> list = inferBatchAsync(images);
//
//    std::for_each(list.begin(), list.end(), [&](const auto &item) {
//        for (const auto &obj: item) {
//            cout << "class_label: " << obj.class_label
//                 << " caption: " << obj.confidence
//                 << " (L T R B): (" << obj.left
//                 << ", " << obj.top << ", "
//                 << obj.right << ", "
//                 << obj.bottom << ")" << endl;
//        }
//    })

// }

//extern "C" __declspec(dllexport) bool
//TensorRT_INIT_ASYNC(const char *engine_file, const float confidence, const float nms, const int batch) {
//    return initBatchAsync(engine_file, confidence, nms, batch);
//}

//static std::vector<yolo::Box> g_boxes;

//extern "C" __declspec(dllexport) int
//TensorRT_INFER_NUM_ASYNC(uchar *image, int width, int height) {
//    g_boxes = inferBatchAsync(image, width, height);
//    return g_boxes.size();
//}
//
//extern "C" __declspec(dllexport) int
//GET_LISTBOX_DATA(int boxLabel,
//                 float *left,
//                 float *top,
//                 float *right,
//                 float *bottom,
//                 float *confidence,
//                 int *classLabel) {
//    if (boxLabel < 0 || boxLabel >= g_boxes.size()) {
//        return -1;
//    }
//    const yolo::Box &box = g_boxes[boxLabel];
//    *left = box.left;
//    *top = box.top;
//    *right = box.right;
//    *bottom = box.bottom;
//    *confidence = box.confidence;
//    *classLabel = box.class_label;
//    return 0;
//}
//
//extern "C" __declspec(dllexport) void
//END_GET_LISTBOX_DATA() {
//    g_boxes.clear();
//}
