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
cudaStream_t cudaStream1;

static vector<cv::Point> xywhr2xyxyxyxy(const obb::Box &box) {
    float cos_value = std::cos(box.angle);
    float sin_value = std::sin(box.angle);

    float w_2 = box.width / 2, h_2 = box.height / 2;
    float vec1_x = w_2 * cos_value, vec1_y = w_2 * sin_value;
    float vec2_x = -h_2 * sin_value, vec2_y = h_2 * cos_value;

    vector<cv::Point> corners;
    corners.push_back(cv::Point(box.center_x + vec1_x + vec2_x, box.center_y + vec1_y + vec2_y));
    corners.push_back(cv::Point(box.center_x + vec1_x - vec2_x, box.center_y + vec1_y - vec2_y));
    corners.push_back(cv::Point(box.center_x - vec1_x - vec2_x, box.center_y - vec1_y - vec2_y));
    corners.push_back(cv::Point(box.center_x - vec1_x + vec2_x, box.center_y - vec1_y + vec2_y));

    return corners;
}

void syncInferObb() {
    cudaStreamCreate(&cudaStream1);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.2, 0.5, cudaStream1);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->obb_forward(yrImage, cudaStream1);
    }

    trt_timer::Timer timer;
    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);

    timer.start(cudaStream1);
    auto objs = yolo->obb_forward(image, cudaStream1);
    timer.stop("batch one");

    std::string windowName = "Image Window";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    int width_ = 1024;
    int height = 640;
    cv::resizeWindow(windowName, width_, height);
    for (auto &obj: objs) {
        uint8_t b = 255, g = 0, r = 255;
        auto corners = xywhr2xyxyxyxy(obj);
        cv::polylines(mat, vector<vector<cv::Point> >{corners}, true, cv::Scalar(b, g, r), 2, 16);

        auto name = obj.class_label;
        auto caption = cv::format("%i %.2f", name, obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(mat, cv::Point(corners[0].x - 3, corners[0].y - 33),
                      cv::Point(corners[0].x - 3 + width, corners[0].y), cv::Scalar(b, g, r), -1);
        cv::putText(mat, caption, cv::Point(corners[0].x - 3, corners[0].y - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imshow(windowName, mat);
    cv::waitKey(0);
}

void syncInfer() {
    cudaStreamCreate(&cudaStream1);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.2, 0.4, cudaStream1);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->forward(yrImage, cudaStream1);
    }

    std::string windowName = "Image Window";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    int width_ = 1024;
    int height = 640;
    cv::resizeWindow(windowName, width_, height);

    trt_timer::Timer timer;
    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);
    timer.start(cudaStream1);
    auto objs = yolo->forward(image, cudaStream1);
    timer.stop("batch one");
    for (auto &obj: objs) {
        cout << "class_label: " << obj.class_label << " caption: " << obj.confidence << " (L T R B): (" << obj.left
                << ", "
                << obj.top << ", " << obj.right << ", " << obj.bottom << ")" << endl;

        rectangle(mat, cv::Point(static_cast<int>(obj.left), static_cast<int>(obj.top)),
                  cv::Point(static_cast<int>(obj.right), static_cast<int>(obj.bottom)),
                  cv::Scalar(255, 0, 255), 2);

        auto name = obj.class_label;
        auto caption = cv::format("%i %.2f", name, obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 1, nullptr).width + 10;
        rectangle(mat, cv::Point(static_cast<int>(obj.left) - 3, static_cast<int>(obj.top) - 33),
                  cv::Point(static_cast<int>(obj.left) + width, static_cast<int>(obj.top)), cv::Scalar(255, 0, 255),
                  -1);
        putText(mat, caption, cv::Point(static_cast<int>(obj.left), static_cast<int>(obj.top) - 5), 0, 1,
                cv::Scalar::all(0), 1,
                16);
    }
    cv::imshow(windowName, mat);
    cv::waitKey(0);
}

// ��������ƺ���
void draw_detection_results(cv::Mat &mat,
                            const detect::BoxArray &objs) {
    for (const auto &obj: objs) {
        // ���˵����Ŷȵļ����
        if (obj.confidence < 0.5) continue;

        // ���Ʊ߽��
        cv::rectangle(mat,
                      cv::Point(static_cast<int>(obj.left), static_cast<int>(obj.top)),
                      cv::Point(static_cast<int>(obj.right), static_cast<int>(obj.bottom)),
                      cv::Scalar(255, 0, 255), // Ʒ��ɫ�߿�
                      2); // �߿�

        // �����ע�ı�
        std::string label = cv::format("%i: %.1f%%",
                                       obj.class_label,
                                       obj.confidence * 100);

        // �����ı��ߴ�
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label,
                                             cv::FONT_HERSHEY_SIMPLEX,
                                             0.5,
                                             1,
                                             &baseline);

        // �����ı�����
        cv::rectangle(mat,
                      cv::Point(static_cast<int>(obj.left),
                                static_cast<int>(obj.top) - text_size.height - 10),
                      cv::Point(static_cast<int>(obj.left) + text_size.width,
                                static_cast<int>(obj.top)),
                      cv::Scalar(255, 0, 255),
                      cv::FILLED);

        // �����ı�
        cv::putText(mat,
                    label,
                    cv::Point(static_cast<int>(obj.left),
                              static_cast<int>(obj.top) - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar::all(0),
                    1,
                    16);
    }
}

void videoDemo() {
    cudaStreamCreate(&cudaStream1);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.1, 0.4, cudaStream1);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->forward(yrImage, cudaStream1);
    }

    // ����Ƶ�������ȳ�����Ϊ�ļ��򿪣�
    cv::VideoCapture cap(config.VIDEO_PATH);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1); // ���ٻ����ӳ�

    // ��ȡ��Ƶ����
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Video Info: " << std::endl;
    std::cout << " - Resolution: " << frame_width << "x" << frame_height << std::endl;
    std::cout << " - Original FPS: " << fps << std::endl;

    cv::Mat mat;
    yolo::Image image;
    trt_timer::Timer timer;

    // ������ʾ����
    cv::namedWindow("YOLO Video Detection",
                    cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow("YOLO Video Detection", 1280, 720);

    // ʱ��ͳ�Ʊ���
    double total_processed_time = 0.0;
    int processed_frames = 0;

    double avg_fps;

    while (true) {
        // ��ȷ��ʱ��ʼ
        double start_time = cv::getTickCount();

        // ��ȡ��Ƶ֡
        cap >> mat;
        if (mat.empty()) break;

        // ת��ΪYOLO�����ʽ
        image = yolo::Image(mat.data, mat.cols, mat.rows);

        // CUDA��������
        timer.start(cudaStream1);
        auto objs = yolo->forward(image, cudaStream1);
        timer.stop("batch one");

        // ���Ƽ����
        draw_detection_results(mat, objs);

        // ���㴦���ʱ
        double process_time = (cv::getTickCount() - start_time) / cv::getTickFrequency();
        total_processed_time += process_time;
        processed_frames++;

        // ���㲢��ʾʵʱFPS
        double current_fps = 1.0 / process_time;
        avg_fps = processed_frames / total_processed_time;

        // ��֡�ϵ�����Ϣ
        cv::putText(mat,
                    cv::format("FPS: %.1f | Avg: %.1f", current_fps, avg_fps),
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 255, 0),
                    2);

        // ��ʾ���
        cv::imshow("YOLO Video Detection", mat);

        // �����˳�����
        const int key = cv::waitKey(1);
        if (key == 27) {
            // ESC���˳�
            break;
        } else if (key == '+' && processed_frames > 0) {
            // ��̬������ʾFPS
            std::cout << "Average FPS reset" << std::endl;
            total_processed_time = 0.0;
            processed_frames = 0;
        }
    }

    // ����ͳ����Ϣ
    std::cout << "Processing completed!" << std::endl;
    std::cout << "Total frames: " << processed_frames << std::endl;
    std::cout << "Average FPS: " << avg_fps << std::endl;

    // �ͷ���Դ
    cap.release();
    cv::destroyAllWindows();
}

bool initSingleCpm(const string &engineFile, float confidence, float nms) {
    cudaStreamCreate(&cudaStream1);
    bool ok = cpmi.start([&engineFile, &confidence, &nms] {
        return yolo::load(engineFile, confidence, nms, cudaStream1);
    }, 1, cudaStream1);
    if (ok) {
        cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
        auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
        for (int i = 0; i < 10; ++i) {
            cpmi.commit(yrImage).get();
        }
        return true;
    } else {
        return false;
    }
}

vector<detect::Box> inferSingleCpm(const cv::Mat &mat) {
    return cpmi.commit(yolo::Image(mat.data, mat.cols, mat.rows)).get();
}

void asyncInfer() {
    Config config;
    if (initSingleCpm(config.MODEL, 0.2, 0.5)) {
        trt_timer::Timer timer;
        const cv::Mat mat = cv::imread(config.TEST_IMG);
        while (true) {
            timer.start();
            inferSingleCpm(mat);
            timer.stop("batch one");
        }
    }
}

void syncInferCls() {
    cudaStreamCreate(&cudaStream1);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.1, 0, cudaStream1);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->cls_forward(yrImage, cudaStream1);
    }

    trt_timer::Timer timer;
    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);
    timer.start(cudaStream1);
    auto objs = yolo->cls_forward(image, cudaStream1);
    timer.stop("batch one");

    std::string windowName = "Image Window";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    int width_ = 1024;
    int height = 640;
    cv::resizeWindow(windowName, width_, height);

    for (int i = 0; i < objs.size(); ++i) {
        auto obj = objs[i];
        cv::putText(mat,
                    std::to_string(obj.class_label) + ": " + std::to_string(obj.confidence).substr(0, 4),
                    cv::Point(10, 30 + i * 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow(windowName, mat);
    cv::waitKey(0);
}

int main() {
    // syncInferCls();
    // syncInferObb();
    // syncInfer();
    // asyncInfer();
    videoDemo();
    return 0;
}
