#include "config.h"
#include "yolo26.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void draw_detections(cv::Mat &image, const yolo26::Detections &detections) {
  for (const auto &detection : detections) {
    const cv::Rect box(cv::Point(static_cast<int>(detection.left),
                                 static_cast<int>(detection.top)),
                       cv::Point(static_cast<int>(detection.right),
                                 static_cast<int>(detection.bottom)));
    cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
    const std::string label = std::to_string(detection.class_id) + " " +
                              cv::format("%.2f", detection.confidence);
    cv::putText(image, label, cv::Point(box.x, std::max(16, box.y - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
  }
}

std::vector<cv::Mat> load_images(const example::Config &config) {
  std::vector<cv::Mat> images;
  images.reserve(config.image_files.size());
  for (const std::string &path : config.image_files) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("cannot read image: " + path);
    }
    images.push_back(std::move(image));
  }
  return images;
}

} // namespace

int main() {
  try {
    const example::Config config;
    yolo26::Detector detector(config.engine_file, config.confidence_threshold,
                              config.gpu_device);
    std::vector<cv::Mat> images = load_images(config);

    std::vector<yolo26::Image> inputs;
    inputs.reserve(images.size());
    for (const cv::Mat &image : images) {
      inputs.emplace_back(image.data, image.cols, image.rows, image.step);
    }

    const std::vector<yolo26::Detections> results = detector.predict(inputs);
    const yolo26::Timing &timing = detector.last_timing();
    std::cout << std::fixed << std::setprecision(3) << "N=" << images.size()
              << ", preprocess=" << timing.preprocess_ms
              << " ms, inference=" << timing.inference_ms
              << " ms, postprocess=" << timing.postprocess_ms
              << " ms, total=" << timing.total_ms << " ms\n";

    if (config.save_images) {
      fs::create_directories(config.output_directory);
    }
    for (std::size_t index = 0; index < results.size(); ++index) {
      if (config.print_detections) {
        for (const auto &detection : results[index]) {
          std::cout << "image=" << index << " class=" << detection.class_id
                    << " score=" << detection.confidence << " box=["
                    << detection.left << ", " << detection.top << ", "
                    << detection.right << ", " << detection.bottom << "]\n";
        }
      }
      if (config.save_images) {
        draw_detections(images[index], results[index]);
        const fs::path source(config.image_files[index]);
        const fs::path output =
            fs::path(config.output_directory) /
            (source.stem().string() + "-result" + source.extension().string());
        if (!cv::imwrite(output.string(), images[index])) {
          throw std::runtime_error("cannot write image: " + output.string());
        }
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
