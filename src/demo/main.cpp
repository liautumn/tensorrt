#include <opencv2/opencv.hpp>
#include <filesystem>
#include "cpm.h"
#include "infer.h"
#include "yolo.h"
#include "config.h"

using namespace std;
namespace fs = std::filesystem;

yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

void batch_inference() {
    Config config;
    vector<cv::Mat> images{
            cv::imread(config.TEST_IMG),
            cv::imread(config.TEST_IMG),
            cv::imread(config.TEST_IMG)
    };
    auto yolo = yolo::load(config.MODEL,
                           yolo::Type::V8);
    if (yolo == nullptr) return;

    vector<yolo::Image> yoloimages(images.size());
    transform(images.begin(), images.end(), yoloimages.begin(), cvimg);
    auto batched_result = yolo->forwards(yoloimages);
    for (int ib = 0; ib < (int) batched_result.size(); ++ib) {
        auto &objs = batched_result[ib];
        auto &image = images[ib];
        for (auto &obj: objs) {
            uint8_t b, g, r;
            tie(b, g, r) = yolo::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                          cv::Scalar(b, g, r), 5);

            auto name = config.cocolabels[obj.class_label];
            auto caption = cv::format("%s %.2f", name, obj.confidence);
            int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                          cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                        16);
        }
        printf("Save result to Result.jpg, %d objects\n", (int) objs.size());
        cv::imwrite(cv::format("Result%d.jpg", ib), image);
    }
}

//void single_inference() {
//    Config config;
//    cv::Mat image = cv::imread("1.jpg");
//    auto yolo = yolo::load(config.MODEL, yolo::Type::V8Seg);
//    if (yolo == nullptr) return;
//
//    auto objs = yolo->forward(cvimg(image));
//    int i = 0;
//    for (auto &obj: objs) {
//        uint8_t b, g, r;
//        tie(b, g, r) = yolo::random_color(obj.class_label);
//        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
//                      cv::Scalar(b, g, r), 5);
//
//        auto name = config.cocolabels[obj.class_label];
//        auto caption = cv::format("%s %.2f", name, obj.confidence);
//        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
//        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
//                      cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
//        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
//
//        if (obj.seg) {
//            cv::imwrite(cv::format("%d_mask.jpg", i),
//                        cv::Mat(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data));
//            i++;
//        }
//    }
//
//    printf("Save result to Result.jpg, %d objects\n", (int) objs.size());
//    cv::imwrite("Result.jpg", image);
//}

std::vector<std::string> get_image_paths(const std::string &directory) {
    std::vector<std::string> image_paths;

    try {
        for (const auto &entry: fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                // Add additional checks here if you want to filter specific image formats
                image_paths.push_back(path);
            }
        }
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Error reading directory: " << e.what() << std::endl;
    }

    return image_paths;
}


void syncInfer() {
    Config config;
    auto yolo = yolo::load(config.MODEL,
                           yolo::Type::V8, 0.2, 0.45);
    if (yolo == nullptr) return;

    trt::Timer timer;

    // 创建一个窗口
    std::string windowName = "Image Window";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    // 调整窗口大小
    int width_ = 1024;  // 设置窗口宽度
    int height = 640; // 设置窗口高度
    cv::resizeWindow(windowName, width_, height);

    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);

    timer.start();
    auto objs = yolo->forward(image);

    vector<cv::Rect> bboxes;
    vector<float> scores;
    vector<int> labels;
    vector<int> indices;

    for (const auto &item: objs) {

        cout << "class_label: " << item.class_label << " caption: " << item.confidence << " (L T R B): (" << item.left
             << ", "
             << item.top << ", " << item.right << ", " << item.bottom << ")" << endl;

        cv::Rect_<float> bbox;
        bbox.x = item.left;
        bbox.y = item.top;
        bbox.width = item.right - item.left;
        bbox.height = item.bottom - item.top;
        bboxes.push_back(bbox);

        labels.push_back(item.class_label);
        scores.push_back(item.confidence);
    }

    cv::dnn::NMSBoxes(bboxes, scores, 0.2, 0.45, indices);

    timer.stop("batch one");

    for (auto &i: indices) {

        double left = bboxes[i].x;
        double top = bboxes[i].y;
        double right = bboxes[i].x + bboxes[i].width;
        double bottom = bboxes[i].y + bboxes[i].height;

        cout << "class_label==============: " << labels[i] << " caption: " << scores[i] << " (L T R B): (" << left
             << ", "
             << top << ", " << right << ", " << bottom << ")" << endl;

        uint8_t b, g, r;
        tie(b, g, r) = yolo::random_color(labels[i]);
        cv::rectangle(mat, cv::Point(left, top), cv::Point(right, bottom),
                      cv::Scalar(b, g, r), 5);

        auto name = config.cocolabels[labels[i]];
        auto caption = cv::format("%s %.2f", name, scores[i]);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(mat, cv::Point(left - 3, top - 33),
                      cv::Point(left + width, top), cv::Scalar(b, g, r), -1);
        cv::putText(mat, caption, cv::Point(left, top - 5), 0, 1, cv::Scalar::all(0), 2,
                    16);
    }
//    for (auto &obj: objs) {
//        cout << "class_label: " << obj.class_label << " caption: " << obj.confidence << " (L T R B): (" << obj.left
//             << ", "
//             << obj.top << ", " << obj.right << ", " << obj.bottom << ")" << endl;
//
//        uint8_t b, g, r;
//
//        tie(b, g, r) = yolo::random_color(obj.class_label);
//        cv::rectangle(mat, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
//                      cv::Scalar(b, g, r), 5);
//
//        auto name = config.cocolabels[obj.class_label];
//        auto caption = cv::format("%s %.2f", name, obj.confidence);
//        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
//        cv::rectangle(mat, cv::Point(obj.left - 3, obj.top - 33),
//                      cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
//        cv::putText(mat, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
//                    16);
//
//    }
    cv::imshow(windowName, mat);  // 显示帧
    cv::waitKey(0);

}

void asyncInfer() {
    Config config;
    cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

    bool ok = cpmi.start([config] {
        return yolo::load(config.MODEL, yolo::Type::V8, 0.25, 0.7);
    });
    if (!ok) return;

    const cv::Mat mat = cv::imread(config.TEST_IMG);
    const auto image = yolo::Image(mat.data, mat.cols, mat.rows);

    trt::Timer timer;

//    while (true) {
    timer.start();
    const auto objs = cpmi.commit(image).get();
    for (auto &obj: objs) {
        const auto name = obj.class_label;
        cout << "class_label: " << name << " caption: " << obj.confidence << " (L T R B): (" << obj.left << ", "
             << obj.top << ", " << obj.right << ", " << obj.bottom << ")" << endl;
    }
    timer.stop("batch one");
//    }
}

void asyncInferVedio() {
    Config config;
    cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

    bool ok = cpmi.start([config] {
        return yolo::load(config.MODEL, yolo::Type::V8, 0.2, 0.5);
    });
    if (!ok) return;

    cv::VideoCapture cap(0);  // 打开默认摄像头
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Unable to open the camera" << std::endl;
        return;
    }
    cv::Mat frame;
    trt::Timer timer;

    vector<cv::Rect> bboxes;
    vector<float> scores;
    vector<int> labels;
    vector<int> indices;

    while (true) {

        bboxes.clear();
        scores.clear();
        labels.clear();
        indices.clear();

        cap >> frame;  // 从摄像头读取新的帧
        if (frame.empty()) {
            std::cerr << "ERROR: Couldn't grab a frame" << std::endl;
            break;
        }
        const auto image = yolo::Image(frame.data, frame.cols, frame.rows);
        timer.start();
        const auto objs = cpmi.commit(image).get();

        for (const auto &item: objs) {
            cv::Rect_<float> bbox;
            bbox.x = item.left;
            bbox.y = item.top;
            bbox.width = item.right - item.left;
            bbox.height = item.bottom - item.top;
            bboxes.push_back(bbox);

            labels.push_back(item.class_label);
            scores.push_back(item.confidence);
        }

        cv::dnn::NMSBoxes(bboxes, scores, 0.3, 0.1, indices);
        timer.stop("batch 1");

        for (auto &i: indices) {

            double left = bboxes[i].x;
            double top = bboxes[i].y;
            double right = bboxes[i].x + bboxes[i].width;
            double bottom = bboxes[i].y + bboxes[i].height;

            uint8_t b, g, r;
            tie(b, g, r) = yolo::random_color(labels[i]);
            cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom),
                          cv::Scalar(b, g, r), 5);

            auto name = config.cocolabels[labels[i]];
            auto caption = cv::format("%s %.2f", name, scores[i]);
            int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(frame, cv::Point(left - 3, top - 33),
                          cv::Point(left + width, top), cv::Scalar(b, g, r), -1);
            cv::putText(frame, caption, cv::Point(left, top - 5), 0, 1, cv::Scalar::all(0), 2,
                        16);
        }
        cv::imshow("yolov8", frame);  // 显示帧
        if (cv::waitKey(1) == 27) break;  // 按 'ESC' 键退出
    }
    cap.release();  // 关闭摄像头
    cv::destroyAllWindows();  // 关闭所有OpenCV窗口
}

int main() {
//    asyncInferVedio();
    syncInfer();
//    asyncInfer();
//    batch_inference();
    return 0;
}
