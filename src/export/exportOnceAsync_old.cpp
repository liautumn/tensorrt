#include <opencv2/opencv.hpp>
#include "infer.hpp"
#include "cpm.hpp"
#include "yolo.hpp"

using namespace std;

static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
static std::vector<yolo::Box> g_boxes;

bool initAsyncOld(const string &engine_file, const float &confidence, const float &nms) {
    bool ok = cpmi.start([&engine_file, &confidence, &nms] {
        return yolo::load(engine_file, yolo::Type::V8, confidence, nms);
    }, 1);
    if (!ok) {
        cout << "================================= TensorRT INIT FAIL =================================" << endl;
        return false;
    } else {
        cout << "================================= TensorRT INIT SUCCESS =================================" << endl;
        return true;
    }
}

vector<yolo::Box> inferAsyncOld(uchar *image, const int &width, const int &height) {
    trt::Timer timer;
    auto img = yolo::Image(image, width, height);
    timer.start();
    auto objs = cpmi.commit(img).get();
    timer.stop("batch 1");
    return objs;
}

extern "C" __declspec(dllexport) bool
TensorRT_INIT_ASYNC_OLD(const char *engine_file, const float confidence, const float nms) {
    return initAsyncOld(engine_file, confidence, nms);
}

extern "C" __declspec(dllexport) int
TensorRT_INFER_NUM_ASYNC_OLD(uchar *image, int width, int height) {
    g_boxes = inferAsyncOld(image, width, height);
    return g_boxes.size();
}

extern "C" __declspec(dllexport) int
GET_LISTBOX_DATA_OLD(int boxLabel,
                 float *left,
                 float *top,
                 float *right,
                 float *bottom,
                 float *confidence,
                 int *classLabel) {
    if (boxLabel >= 0 && boxLabel < g_boxes.size()) {
        const yolo::Box &box = g_boxes[boxLabel];
        *left = box.left;
        *top = box.top;
        *right = box.right;
        *bottom = box.bottom;
        *confidence = box.confidence;
        *classLabel = box.class_label;
        return 0;
    } else {
        return -1;
    }
}

extern "C" __declspec(dllexport) void
END_GET_LISTBOX_DATA_OLD() {
    g_boxes.clear();
}
