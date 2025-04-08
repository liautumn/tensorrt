#ifndef YOLO_H
#define YOLO_H

#include <future>
#include <memory>
#include <string>
#include <vector>

namespace yolo
{
    using namespace std;

    struct Box
    {
        float left, top, right, bottom, confidence;
        int class_label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int class_label)
            : left(left),
              top(top),
              right(right),
              bottom(bottom),
              confidence(confidence),
              class_label(class_label)
        {
        }
    };


    struct Image
    {
        const void* bgrptr = nullptr;
        int width = 0, height = 0;

        Image() = default;

        Image(const void* bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height)
        {
        }
    };

    typedef vector<Box> BoxArray;

    class Infer
    {
    public:
        virtual BoxArray forward(const Image& image, void* stream = nullptr) = 0;

        virtual vector<BoxArray> forwards(const vector<Image>& images,
                                          void* stream = nullptr) = 0;
    };

    shared_ptr<Infer> load(const string& engine_file,
                           float confidence_threshold = 0.2f,
                           float nms_threshold = 0.5f,
                           void* stream = nullptr);
}; // namespace yolo

#endif  // YOLO_H
