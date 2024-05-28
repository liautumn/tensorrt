#include <opencv2/opencv.hpp>
#include "infer.hpp"
#include "cpm.hpp"
#include "yolo.hpp"

using namespace std;

const int width = 1920;
const int height = 1200;

cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

bool initAsync(const string &engine_file) {
    bool ok = cpmi.start([&engine_file] {
        return yolo::load(engine_file, yolo::Type::V8);
    }, 1);
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
    auto img = yolo::Image(image, width, height);
    timer.start();
    auto objs = cpmi.commit(img).get();
    timer.stop("BATCH1");
    return objs;
}

extern "C" __declspec(dllexport) bool TensorRT_INIT_ASYNC(const char *engine_file) {
    return initAsync(engine_file);
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