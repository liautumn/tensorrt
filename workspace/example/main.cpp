#include "config.h"

static void draw_pose(cv::Mat &image, const vector<cv::Point3f> &keypoints) {
    vector<cv::Scalar> pose_palette = {
        {255, 128, 0}, {255, 153, 51}, {255, 178, 102}, {230, 230, 0}, {255, 153, 255},
        {153, 204, 255}, {255, 102, 255}, {255, 51, 255}, {102, 178, 255}, {51, 153, 255},
        {255, 153, 153}, {255, 102, 102}, {255, 51, 51}, {153, 255, 153}, {102, 255, 102},
        {51, 255, 51}, {0, 255, 0}, {0, 0, 255}, {255, 0, 0}, {255, 255, 255}
    };

    vector<cv::Point> skeleton = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
        {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1},
        {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}
    };
    // 16 0 9
    //
    vector<cv::Scalar> limb_color = {
        pose_palette[9], pose_palette[9], pose_palette[9], pose_palette[9], pose_palette[7],
        pose_palette[7], pose_palette[7], pose_palette[0], pose_palette[0], pose_palette[0],
        pose_palette[0], pose_palette[0], pose_palette[16], pose_palette[16], pose_palette[16],
        pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16]
    };

    vector<cv::Scalar> kpt_color = {
        pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16],
        pose_palette[0], pose_palette[0], pose_palette[0], pose_palette[0], pose_palette[0],
        pose_palette[0], pose_palette[9], pose_palette[9], pose_palette[9], pose_palette[9],
        pose_palette[9], pose_palette[9]
    };

    for (int i = 0; i < keypoints.size(); ++i) {
        auto &keypoint = keypoints[i];
        if (keypoint.z < 0.5)
            continue;
        if (keypoint.x != 0 && keypoint.y != 0)
            cv::circle(image, cv::Point(keypoint.x, keypoint.y), 5, kpt_color[i], -1, cv::LINE_AA);
    }

    for (int i = 0; i < skeleton.size(); ++i) {
        auto &index = skeleton[i];
        auto &pos1 = keypoints[index.x];
        auto &pos2 = keypoints[index.y];

        if (pos1.z < 0.5 || pos2.z < 0.5)
            continue;

        if (pos1.x == 0 || pos1.y == 0 || pos2.x == 0 || pos2.y == 0)
            continue;

        cv::line(image, cv::Point(pos1.x, pos1.y), cv::Point(pos2.x, pos2.y), limb_color[i], 2, cv::LINE_AA);
    }
}

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
    cudaStreamCreate(&cudaStream);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.2, 0.5, config.GPU_DEVICE, cudaStream);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->obb_forward(yrImage, cudaStream);
    }

    trt_timer::Timer timer;
    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);

    timer.start(cudaStream);
    auto objs = yolo->obb_forward(image, cudaStream);
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

void syncInferDetect() {
    cudaStreamCreate(&cudaStream);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.2, 0.4, config.GPU_DEVICE, cudaStream);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->detect_forward(yrImage, cudaStream);
    }

    trt_timer::Timer timer;
    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);
    timer.start(cudaStream);
    auto objs = yolo->detect_forward(image, cudaStream);
    timer.stop("batch one");

    std::string windowName = "Image Window";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    int width_ = 1024;
    int height = 640;
    cv::resizeWindow(windowName, width_, height);
    for (auto &obj: objs) {
        rectangle(mat, cv::Point(static_cast<int>(obj.left), static_cast<int>(obj.top)),
                  cv::Point(static_cast<int>(obj.right), static_cast<int>(obj.bottom)),
                  cv::Scalar(255, 0, 255), 2);
        auto caption = cv::format("%i %.2f", obj.class_label, obj.confidence);
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

static void draw_mask(cv::Mat &image, seg::Box &obj, cv::Scalar &color) {
    // compute IM
    float scale_x = 640 / static_cast<float>(image.cols);
    float scale_y = 640 / static_cast<float>(image.rows);
    float scale = std::min(scale_x, scale_y);
    float ox = -scale * image.cols * 0.5 + 640 * 0.5 + scale * 0.5 - 0.5;
    float oy = -scale * image.rows * 0.5 + 640 * 0.5 + scale * 0.5 - 0.5;
    cv::Mat M = (cv::Mat_<float>(2, 3) << scale, 0, ox, 0, scale, oy);

    cv::Mat IM;
    cv::invertAffineTransform(M, IM);

    cv::Mat mask_map = cv::Mat::zeros(cv::Size(160, 160), CV_8UC1);
    cv::Mat small_mask(obj.seg->height, obj.seg->width, CV_8UC1, obj.seg->data);
    cv::Rect roi(obj.seg->left, obj.seg->top, obj.seg->width, obj.seg->height);
    small_mask.copyTo(mask_map(roi));
    cv::resize(mask_map, mask_map, cv::Size(640, 640)); // 640x640
    cv::threshold(mask_map, mask_map, 128, 1, cv::THRESH_BINARY);

    cv::Mat mask_resized;
    cv::warpAffine(mask_map, mask_resized, IM, image.size(), cv::INTER_LINEAR);

    // create color mask
    cv::Mat colored_mask = cv::Mat::ones(image.size(), CV_8UC3);
    colored_mask.setTo(color);

    cv::Mat masked_colored_mask;
    cv::bitwise_and(colored_mask, colored_mask, masked_colored_mask, mask_resized);

    // create mask indices
    cv::Mat mask_indices;
    cv::compare(mask_resized, 1, mask_indices, cv::CMP_EQ);

    cv::Mat image_masked, colored_mask_masked;
    image.copyTo(image_masked, mask_indices);
    masked_colored_mask.copyTo(colored_mask_masked, mask_indices);

    // weighted sum
    cv::Mat result_masked;
    cv::addWeighted(image_masked, 0.6, colored_mask_masked, 0.4, 0, result_masked);

    // copy result to image
    result_masked.copyTo(image, mask_indices);
}

void syncInferSeg() {
    cudaStreamCreate(&cudaStream);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.1, 0.4, config.GPU_DEVICE, cudaStream);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->seg_forward(yrImage, cudaStream);
    }

    trt_timer::Timer timer;
    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);
    timer.start(cudaStream);
    auto boxes = yolo->seg_forward(image, cudaStream);
    timer.stop("batch one");

    std::string windowName = "Image Window";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    int width_ = 640;
    int height = 640;
    cv::resizeWindow(windowName, width_, height);

    for (auto &obj: boxes) {
        cv::Scalar color(255, 0, 255);
        if (obj.seg) {
            draw_mask(mat, obj, color);
        }
    }
    for (auto &obj: boxes) {
        cv::rectangle(mat, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(255, 0, 255), 5);
        auto caption = cv::format("%i %.2f", obj.class_label, obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(mat, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top),
                      cv::Scalar(255, 0, 255), -1);
        cv::putText(mat, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imshow(windowName, mat);
    cv::waitKey(0);
    // int i = 0;
    // for (auto &obj: boxes) {
    //     cv::rectangle(mat, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
    //                   cv::Scalar(255, 0, 255), 5);
    //
    //     auto caption = cv::format("%i %.2f", obj.class_label, obj.confidence);
    //     int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    //     cv::rectangle(mat, cv::Point(obj.left - 3, obj.top - 33),
    //                   cv::Point(obj.left + width, obj.top), cv::Scalar(255, 0, 255), -1);
    //     cv::putText(mat, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    //
    //     if (obj.seg) {
    //         cv::imwrite(cv::format("%d_mask.jpg", i),
    //                     cv::Mat(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data));
    //         i++;
    //     }
    // }
    // cv::imwrite("Result.jpg", mat);
}

// 检测结果绘制函数
void draw_detection_results(cv::Mat &mat, const obb::BoxArray &objs) {
    for (const auto &obj: objs) {
        // 过滤低置信度的检测结果
        if (obj.confidence < 0.5) continue;

        uint8_t b = 255, g = 0, r = 255;
        auto corners = xywhr2xyxyxyxy(obj);
        cv::polylines(mat, vector<vector<cv::Point> >{corners}, true, cv::Scalar(b, g, r), 2, 16);

        auto name = obj.class_label;
        auto caption = cv::format("%i %.2f", name, obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::putText(mat, caption, cv::Point(corners[0].x - 3, corners[0].y - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        cv::rectangle(mat, cv::Point(corners[0].x - 3, corners[0].y - 33),
                      cv::Point(corners[0].x - 3 + width, corners[0].y), cv::Scalar(b, g, r), -1);
    }
}

void videoDemo() {
    cudaStreamCreate(&cudaStream);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.2, 0.4, config.GPU_DEVICE, cudaStream);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->pose_forward(yrImage, cudaStream);
    }

    // 打开视频流（优先尝试作为文件打开）
    cv::VideoCapture cap(config.VIDEO_PATH);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1); // 减少缓冲延迟

    // 获取视频属性
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Video Info: " << std::endl;
    std::cout << " - Resolution: " << frame_width << "x" << frame_height << std::endl;
    std::cout << " - Original FPS: " << fps << std::endl;

    cv::Mat mat;
    yolo::Image image;
    trt_timer::Timer timer;

    // 创建显示窗口
    cv::namedWindow("YOLO Video Detection",
                    cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow("YOLO Video Detection", 1280, 720);

    // 时间统计变量
    double total_processed_time = 0.0;
    int processed_frames = 0;

    double avg_fps;

    while (true) {
        // 精确计时开始
        double start_time = cv::getTickCount();

        // 读取视频帧
        cap >> mat;
        if (mat.empty()) break;

        // 转换为YOLO处理格式
        image = yolo::Image(mat.data, mat.cols, mat.rows);

        // CUDA加速推理
        timer.start(cudaStream);
        auto objs = yolo->pose_forward(image, cudaStream);
        timer.stop("batch one");

        // 绘制检测结果
        // draw_detection_results(mat, objs);

        for (auto &obj: objs) {
            cv::rectangle(mat, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(255, 0, 255),
                          5);

            auto caption = cv::format("person %.2f", obj.confidence);
            int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(mat, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top),
                          cv::Scalar(255, 0, 255), -1);
            cv::putText(mat, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            draw_pose(mat, obj.keypoints);
        }

        // 计算处理耗时
        double process_time = (cv::getTickCount() - start_time) / cv::getTickFrequency();
        total_processed_time += process_time;
        processed_frames++;

        // 计算并显示实时FPS
        double current_fps = 1.0 / process_time;
        avg_fps = processed_frames / total_processed_time;

        // 在帧上叠加信息
        cv::putText(mat,
                    cv::format("FPS: %.1f | Avg: %.1f", current_fps, avg_fps),
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 255, 0),
                    2);

        // 显示结果
        cv::imshow("YOLO Video Detection", mat);

        // 处理退出按键
        const int key = cv::waitKey(1);
        if (key == 27) {
            // ESC键退出
            break;
        } else if (key == '+' && processed_frames > 0) {
            // 动态调整显示FPS
            std::cout << "Average FPS reset" << std::endl;
            total_processed_time = 0.0;
            processed_frames = 0;
        }
    }

    // 最终统计信息
    std::cout << "Processing completed!" << std::endl;
    std::cout << "Total frames: " << processed_frames << std::endl;
    std::cout << "Average FPS: " << avg_fps << std::endl;

    // 释放资源
    cap.release();
    cv::destroyAllWindows();
}

bool initSingleCpm(const int gpu_device, const string &engineFile, float confidence, float nms) {
    cudaStreamCreate(&cudaStream);
    bool ok = cpmi.start([ &engineFile, &confidence, &nms,&gpu_device] {
        return yolo::load(engineFile, confidence, nms, gpu_device, cudaStream);
    }, 1, cudaStream);
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

void asyncInferDetect() {
    Config config;
    if (initSingleCpm(config.GPU_DEVICE, config.MODEL, 0.2, 0.5)) {
        trt_timer::Timer timer;
        const cv::Mat mat = cv::imread(config.TEST_IMG);
        while (true) {
            timer.start(cudaStream);
            inferSingleCpm(mat);
            timer.stop("batch one");
        }
    }
}

void syncInferCls() {
    cudaStreamCreate(&cudaStream);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.1, 0, config.GPU_DEVICE, cudaStream);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->cls_forward(yrImage, cudaStream);
    }

    trt_timer::Timer timer;
    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);
    timer.start(cudaStream);
    auto objs = yolo->cls_forward(image, cudaStream);
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

void syncInferPose() {
    cudaStreamCreate(&cudaStream);

    Config config;
    auto yolo = yolo::load(config.MODEL, 0.2, 0.4, config.GPU_DEVICE, cudaStream);
    if (yolo == nullptr) return;

    cv::Mat yrMat = cv::Mat(1200, 1920, CV_8UC3);
    auto yrImage = yolo::Image(yrMat.data, yrMat.cols, yrMat.rows);
    for (int i = 0; i < 10; ++i) {
        auto objs = yolo->pose_forward(yrImage, cudaStream);
    }

    trt_timer::Timer timer;
    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);
    timer.start(cudaStream);
    auto objs = yolo->pose_forward(image, cudaStream);
    timer.stop("batch one");

    std::string windowName = "Image Window";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    int width_ = 1024;
    int height = 640;
    cv::resizeWindow(windowName, width_, height);
    for (auto &obj: objs) {
        // cv::rectangle(mat, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(255, 0, 255), 5);
        //
        // auto caption = cv::format("person %.2f", obj.confidence);
        // int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        // cv::rectangle(mat, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top),
        //               cv::Scalar(255, 0, 255), -1);
        // cv::putText(mat, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        draw_pose(mat, obj.keypoints);
    }
    cv::imshow(windowName, mat);
    cv::waitKey(0);
}

int main() {
    // syncInferPose();
    // syncInferSeg();
    // syncInferCls();
    // syncInferObb();
    // syncInferDetect();
    // asyncInferDetect();
    videoDemo();
    return 0;
}
