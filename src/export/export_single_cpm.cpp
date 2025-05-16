#include <opencv2/opencv.hpp>
#include "cuda_runtime_api.h"
#include "infer.h"
#include "cpm.h"
#include "yolo.h"

using namespace std;

static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

cudaStream_t stream; // 定义 CUDA Stream 对象

bool initSingleCpm(const string &engineFile, float confidence, float nms) {
    // 创建 Stream
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        printf("Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
        return false;
    }
    bool ok = cpmi.start([&engineFile, &confidence, &nms] {
        return yolo::load(engineFile, yolo::Type::V8, confidence, nms);
    }, 1, stream);
    if (!ok) {
        return false;
    }
    return true;
}

vector<yolo::Box> inferSingleCpm(cv::Mat *mat) {
    return cpmi.commit(yolo::Image(mat->data, mat->cols, mat->rows)).get();
}

extern "C" __declspec(dllexport) bool TENSORRT_SINGLE_CPM_INIT(const char *engineFile, float confidence, float nms) {
    return initSingleCpm(engineFile, confidence, nms);
}

extern "C" __declspec(dllexport) void TENSORRT_SINGLE_CPM_INFER(cv::Mat *mat, yolo::Box **result, int *size) {
    // 调用推理函数获取检测框
    std::vector<yolo::Box> boxes = inferSingleCpm(mat);
    // 设置返回的大小
    *size = boxes.size();
    // 为结果分配内存
    *result = new yolo::Box[boxes.size()];
    // 拷贝结果到分配的内存中
    std::copy(boxes.begin(), boxes.end(), *result);
}

extern "C" __declspec(dllexport) void TENSORRT_SINGLE_CPM_DESTROY() {
    cpmi.stop();
    // 销毁 Stream
    cudaError_t err = cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        printf("Failed to destroy CUDA stream: %s\n", cudaGetErrorString(err));
    }
}
