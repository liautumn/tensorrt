#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "yolo.h"
#include "config.h"
#include "cpm.h"
#include "timer.h"

using namespace std;
namespace fs = std::filesystem;

static cpm::Instance<detect::BoxArray, yolo::Image, yolo::Infer> cpmi;
inline cudaStream_t cudaStream;

class Config {
public:
    const int GPU_DEVICE = 0;
    const string MODEL = R"(/home/autumn/Documents/GitHub/tensorrt/workspace/model/engine/yolo11s.dynamic.transd.engine)";
    const string TEST_IMG = R"(/home/autumn/Documents/GitHub/tensorrt/workspace/images/bus.jpg)";
    const string VIDEO_PATH = R"(/home/autumn/Documents/GitHub/tensorrt/workspace/images/002.mp4)";
};

#endif //YOLO_CONFIG_H
