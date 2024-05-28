#include <opencv2/opencv.hpp>
#include "infer.hpp"
#include "cpm.hpp"
#include "yolo.hpp"
#include "interface.h"

using namespace std;

cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

bool initAsync(const string &engine_file,
               const float &confidence,
               const float &nms,
               const int &width,
               const int &height) {
    img_width = width;
    img_height = height;
    bool ok = cpmi.start([&engine_file, &confidence, &nms] {
        return yolo::load(engine_file, yolo::Type::V8, confidence, nms);
    });
    if (!ok) {
        cout << "================================= TensorRT INIT FAIL =================================" << endl;
        return false;
    } else {
        cout << "================================= TensorRT INIT SUCCESS =================================" << endl;
        return true;
    }
}

vector<yolo::Box> inferAsync(uchar *image) {
    trt::Timer timer;
    auto img = yolo::Image(image, img_width, img_height);
    timer.start();
    auto objs = cpmi.commit(img).get();
    timer.stop("batch 1");
    return objs;
}

extern "C" __declspec(dllexport) bool TensorRT_INIT_ASYNC(const char *engine_file,
                                                          const float confidence,
                                                          const float nms,
                                                          const int width,
                                                          const int height) {
    return initAsync(engine_file, confidence, nms, width, height);
}
extern "C" __declspec(dllexport) yolo::Box *TensorRT_INFER_ASYNC(uchar *image, int *size) {
    vector<yolo::Box> boxes = inferAsync(image);
    *size = boxes.size();
    auto *result = new yolo::Box[boxes.size()];
    copy(boxes.begin(), boxes.end(), result);
    return result;
}
extern "C" __declspec(dllexport) void FreeMemory_ASYNC(yolo::Box *ptr) {
    delete[] ptr;
}