#include <opencv2/opencv.hpp>
#include "infer.hpp"
#include "yolo.hpp"
#include "interface.h"

using namespace std;

shared_ptr<yolo::Infer> my_yolo;

bool initSync(const char *engine_file,
              const float confidence,
              const float nms,
              const int width,
              const int height) {
    img_width = width;
    img_height = height;
    my_yolo = yolo::load(engine_file, yolo::Type::V8, confidence, nms);
    if (my_yolo == nullptr) {
        cout << "================================= TensorRT INIT FAIL =================================" << endl;
        return false;
    } else {
        cout << "================================= TensorRT INIT SUCCESS =================================" << endl;
        return true;
    }
}

vector<yolo::Box> inferSync(uchar *image) {
    trt::Timer timer;
    auto img = yolo::Image(image, img_width, img_height);
    timer.start();
    auto objs = my_yolo->forward(img);
    timer.stop("batch 1");
    return objs;
}

extern "C" __declspec(dllexport) bool TensorRT_INIT_SYNC(const char *engine_file,
                                                         const float confidence,
                                                         const float nms,
                                                         const int width,
                                                         const int height) {
    return initSync(engine_file, confidence, nms, width, height);
}
extern "C" __declspec(dllexport) yolo::Box *TensorRT_INFER_SYNC(uchar *image, int *size) {
    vector<yolo::Box> boxes = inferSync(image);
    *size = boxes.size();
    auto *result = new yolo::Box[boxes.size()];
    copy(boxes.begin(), boxes.end(), result);
    return result;
}
extern "C" __declspec(dllexport) void FreeMemory_SYNC(yolo::Box *ptr) {
    delete[] ptr;
}