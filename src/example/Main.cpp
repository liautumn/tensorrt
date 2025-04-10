#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "Yolo.h"
#include "Config.h"
#include "Cpm.h"
#include "Timer.h"

using namespace std;
namespace fs = std::filesystem;

static cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
cudaStream_t cudaStream1;

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

    trt_timer::Timer Timer;
    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);
    Timer.start(cudaStream1);
    auto objs = yolo->forward(image, cudaStream1);
    Timer.stop("batch one");
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
                            const yolo::BoxArray &objs) {
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

    // ��Ƶ�ļ�·������
    const std::string video_path = "D:/autumn/Downloads/001.mp4"; // ȷ��·����ȷ

    // ����Ƶ�������ȳ�����Ϊ�ļ��򿪣�
    cv::VideoCapture cap(video_path);
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
    trt_timer::Timer Timer;

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
        Timer.start(cudaStream1);
        auto objs = yolo->forward(image, cudaStream1);
        Timer.stop("batch one");

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

vector<yolo::Box> inferSingleCpm(const cv::Mat &mat) {
    return cpmi.commit(yolo::Image(mat.data, mat.cols, mat.rows)).get();
}

void asyncInfer() {
    Config config;
    if (initSingleCpm(config.MODEL, 0.2, 0.5)) {
        trt_timer::Timer Timer;
        const cv::Mat mat = cv::imread(config.TEST_IMG);
        while (true) {
            Timer.start();
            inferSingleCpm(mat);
            Timer.stop("batch one");
        }
    }
}

int main() {
    // syncInfer();
    // asyncInfer();
    videoDemo();
    return 0;
}
