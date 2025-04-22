#ifndef EXPORT_COMMON_H
#define EXPORT_COMMON_H
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <driver_types.h>
#include "infer.h"
#include "yolo.h"
#include "cpm.h"
#include "timer.h"

using namespace std;

#ifdef _WIN32
    #define EXPORT_API extern "C" __declspec(dllexport)
#else
    #define EXPORT_API extern "C" __attribute__((visibility("default")))
#endif

static cpm::Instance<detect::BoxArray, yolo::Image, yolo::Infer> cpmi;
static shared_ptr<yolo::Infer> my_yolo;
inline cudaStream_t cudaStream;

#endif //EXPORT_COMMON_H
