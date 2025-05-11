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
    const string DETECT_MODEL = R"(D:\autumn\Documents\GitHub\tensorrt\workspace\model\engine\yolo11s.transd.engine)";
    const string SEG_MODEL = R"(D:\autumn\Documents\GitHub\tensorrt\workspace\model\engine\yolo11s.transd.engine)";
    const string CLS_MODEL = R"(D:\autumn\Documents\GitHub\tensorrt\workspace\model\engine\yolo11s-cls.engine)";
    const string OBB_MODEL = R"(D:\autumn\Documents\GitHub\tensorrt\workspace\model\engine\yolo11s-obb.transd.engine)";
    const string POSE_MODEL = R"(D:\autumn\Documents\GitHub\tensorrt\workspace\model\engine\YOLO11s-pose.transd.engine)";
    const string TEST_IMG = R"(D:\autumn\Documents\GitHub\tensorrt\workspace\images\P0009.jpg)";
    const string VIDEO_PATH = R"(D:\autumn\Documents\GitHub\tensorrt\workspace\images\001.mp4)";
};

#endif //YOLO_CONFIG_H
