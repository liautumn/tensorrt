#ifndef YOLO_H
#define YOLO_H

#include <future>
#include <memory>
#include <string>
#include <vector>
#include "DetectPostprocess.cuh"
#include "ObbPostprocess.cuh"

namespace yolo {
    using namespace std;

    struct Image {
        const void *bgrptr = nullptr;
        int width = 0, height = 0;

        Image() = default;

        Image(const void *bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {
        }
    };

    class Infer {
    public:
        virtual detect::BoxArray forward(const Image &image, void *stream = nullptr) = 0;

        virtual obb::BoxArray obbForward(const Image &image, void *stream = nullptr) = 0;

        virtual vector<detect::BoxArray> forwards(const vector<Image> &images, void *stream = nullptr) = 0;

        virtual vector<obb::BoxArray> obbForwards(const vector<Image> &images, void *stream = nullptr) = 0;
    };

    shared_ptr<Infer> load(const string &engine_file,
                           float confidence_threshold = 0.2f,
                           float nms_threshold = 0.5f,
                           void *stream = nullptr);
}; // namespace yolo

#endif  // YOLO_H
