#include <opencv2/opencv.hpp>
#include "cpm.h"
#include "infer.h"
#include "yolo.h"
#include "config.h"

using namespace std;

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

void syncInfer() {
    Config config;
    auto yolo = yolo::load(config.MODEL,
                           yolo::Type::V8);
    if (yolo == nullptr) return;

    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);

    trt::Timer timer;

    while (true) {
        timer.start();
        auto objs = yolo->forward(image);
        timer.stop("batch 1");
    }
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

    while (true) {
        timer.start();
        const auto objs = cpmi.commit(image).get();
        for (auto &obj: objs) {
            const auto name = obj.class_label;
            cout << "class_label: " << name << " caption: " << obj.confidence << " (L T R B): (" << obj.left << ", "
                 << obj.top << ", " << obj.right << ", " << obj.bottom << ")" << endl;
        }
        timer.stop("batch 1");
    }
}

void asyncInferVedio() {
    Config config;
    cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

    bool ok = cpmi.start([config] {
        return yolo::load(config.MODEL, yolo::Type::V8, 0.25, 0.7);
    });
    if (!ok) return;

    cv::VideoCapture cap(0);  // ��Ĭ������ͷ
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Unable to open the camera" << std::endl;
        return;
    }
    cv::Mat frame;
    trt::Timer timer;
    while (true) {
        cap >> frame;  // ������ͷ��ȡ�µ�֡
        if (frame.empty()) {
            std::cerr << "ERROR: Couldn't grab a frame" << std::endl;
            break;
        }
        const auto image = yolo::Image(frame.data, frame.cols, frame.rows);
        timer.start();
        const auto objs = cpmi.commit(image).get();
        timer.stop("batch 1");
        for (auto &obj: objs) {
            uint8_t b, g, r;
            tie(b, g, r) = yolo::random_color(obj.class_label);
            cv::rectangle(frame, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                          cv::Scalar(b, g, r), 5);

            auto name = config.cocolabels[obj.class_label];
            auto caption = cv::format("%s %.2f", name, obj.confidence);
            int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(frame, cv::Point(obj.left - 3, obj.top - 33),
                          cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(frame, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                        16);
        }
        cv::imshow("yolov8", frame);  // ��ʾ֡
        if (cv::waitKey(1) == 27) break;  // �� 'ESC' ���˳�
    }
    cap.release();  // �ر�����ͷ
    cv::destroyAllWindows();  // �ر�����OpenCV����
}

int main() {
    asyncInferVedio();
//    syncInfer();
//    asyncInfer();
//    batch_inference();
    return 0;
}
