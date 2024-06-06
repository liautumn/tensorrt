#include <opencv2/opencv.hpp>
#include "infer.hpp"
#include "yolo.hpp"

using namespace std;

static shared_ptr<yolo::Infer> my_yolo;

bool initAsync(const string &engine_file) {
    my_yolo = yolo::load(engine_file, yolo::Type::V8);
    if (my_yolo == nullptr) {
        cout << "================================= TensorRT INIT FAIL =================================" << endl;
        return false;
    } else {
        cout << "================================= TensorRT INIT SUCCESS =================================" << endl;
        return true;
    }
}

vector<yolo::Box> inferAsync(uchar *image, int width, int height) {
    trt::Timer timer;
    auto img = yolo::Image(image, width, height);
    timer.start();
    auto objs = my_yolo->forward(img);
    timer.stop("batch 1");
    return objs;
}

extern "C" __declspec(dllexport) bool TensorRT_INIT_SYNC(const char *engine_file) {
    return initAsync(engine_file);
}

static std::vector<yolo::Box> g_boxes;

extern "C" __declspec(dllexport) int TensorRT_INFER_NUM_SYNC(uchar *image, int width, int height) {
    g_boxes = inferAsync(image, width, height);
    return g_boxes.size();
}

extern "C" __declspec(dllexport) int GET_LISTBOX_DATA_SYNC(int boxLabel,
                                                      float *left,
                                                      float *top,
                                                      float *right,
                                                      float *bottom,
                                                      float *confidence,
                                                      int *classLabel) {
    if (boxLabel < 0 || boxLabel >= g_boxes.size()) {
        return -1;
    }
    const yolo::Box &box = g_boxes[boxLabel];
    *left = box.left;
    *top = box.top;
    *right = box.right;
    *bottom = box.bottom;
    *confidence = box.confidence;
    *classLabel = box.class_label;
    return 0;
}

extern "C" __declspec(dllexport) void END_GET_LISTBOX_DATA_SYNC() {
    g_boxes.clear();
}
