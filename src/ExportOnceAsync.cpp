#include "cpm.hpp"
#include "infer.hpp"
#include "iostream"
#include "yolo.hpp"

using namespace std;

cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

extern "C" {
__declspec(dllexport) bool TensorRT_INIT_ASYNC(const char *engine_file, float confidence, float nms) {
    cout << "engine_file: " << engine_file << endl;
    cout << "confidence: " << confidence << endl;
    cout << "nms: " << nms << endl;
    bool ok = cpmi.start([&engine_file, &confidence, &nms] {
        return yolo::load(engine_file, yolo::Type::V8, confidence, nms);
    });
    if (!ok) {
        std::cout << "================================= TensorRT INIT FAIL =================================" <<
                std::endl;
        return false;
    } else {
        std::cout << "================================= TensorRT INIT SUCCESS =================================" <<
                std::endl;
        return true;
    }
}

__declspec(dllexport) yolo::Box *TensorRT_INFER_ASYNC(const unsigned char *image, const int *width, const int *height,
                                                      int *size) {
    trt::Timer timer;
    auto img = yolo::Image(image, *width, *height);
    timer.start();
    auto boxes = cpmi.commit(img).get();
    timer.stop("batch 1");
    *size = static_cast<int>(boxes.size());
    yolo::Box *result = new yolo::Box[boxes.size()];
    std::copy(boxes.begin(), boxes.end(), result);
    return result;
}

__declspec(dllexport) void FreeMemory_ASYNC(yolo::Box *ptr) {
    delete[] ptr;
}
}
