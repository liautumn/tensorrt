// #include <opencv2/core/hal/interface.h>
//
// #include "cpm.hpp"
// #include "infer.hpp"
// #include "iostream"
// #include "yolo.hpp"
//
// using namespace std;
//
// cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
//
// extern "C" {
// __declspec(dllexport) bool TensorRT_INIT_ASYNC(const char *engine_file, float confidence, float nms) {
//     cout << "engine_file: " << engine_file << endl;
//     cout << "confidence: " << confidence << endl;
//     cout << "nms: " << nms << endl;
//     bool ok = cpmi.start([&engine_file, &confidence, &nms] {
//         return yolo::load(engine_file, yolo::Type::V8, confidence, nms);
//     });
//     if (!ok) {
//         std::cout << "================================= TensorRT INIT FAIL =================================" <<
//                 std::endl;
//         return false;
//     } else {
//         std::cout << "================================= TensorRT INIT SUCCESS =================================" <<
//                 std::endl;
//         return true;
//     }
// }
//
// __declspec(dllexport) yolo::Box *TensorRT_INFER_ASYNC(const uchar *image, int *size) {
//     // trt::Timer timer;
//     // auto img = yolo::Image(image, 1920, 1200);
//     // timer.start();
//     // auto boxes = cpmi.commit(img).get();
//     // timer.stop("batch 1");
//     // *size = static_cast<int>(boxes.size());
//     // yolo::Box *result = new yolo::Box[boxes.size()];
//     // std::copy(boxes.begin(), boxes.end(), result);
//     //
//     // for (auto &obj: boxes) {
//     //     auto name = obj.class_label;
//     //     cout << "class_label: " << name << " caption: " << obj.confidence << " (L T R D B): (" << obj.left << ", "
//     //             << obj.top << ", " << obj.right << ", " << obj.bottom << ")" <<
//     //             endl;
//     // }
//
//
//     *size = 2; // 示例：返回两个 Box
//     yolo::Box *result = new yolo::Box[*size];
//
//     result[0].left = 394.539f;
//     result[0].top = -22.5312f;
//     result[0].right = 560.242f;
//     result[0].bottom = 1227.16f;
//     result[0].class_label = 0;
//
//     result[1].left = 905.828f;
//     result[1].top = 1.84381f;
//     result[1].right = 1168.8f;
//     result[1].bottom = 1180.28f;
//     result[1].class_label = 1;
//
//     // 输出调试信息
//     for (int i = 0; i < *size; ++i) {
//         std::cout << "Box " << i << ": left=" << result[i].left << ", top=" << result[i].top
//                 << ", right=" << result[i].right << ", bottom=" << result[i].bottom
//                 << ", class_label=" << result[i].class_label << std::endl;
//     }
//
//     return result;
// }
//
// __declspec(dllexport) void FreeMemory_ASYNC(yolo::Box *ptr) {
//     delete[] ptr;
// }
// }
