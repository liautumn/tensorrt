// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

#include "batch.h"
#include "image.h"
#include "inference.h"
#include "model.h"
#include "preprocess.h"
#include "result.h"
#include "timer.h"
#include "validator.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{

struct DetectionRun
{
    Results results;
    double preprocessMilliseconds;
    float inferenceMilliseconds;
    double postprocessMilliseconds;
};

// 图片目录和视频帧共同使用的预处理、推理及后处理流程。
DetectionRun runDetection(
    Model& model,
    Images const& images,
    Batch const& batch,
    float confidenceThreshold,
    trt_timer::Timer& inferenceTimer)
{
    setBatchSize(model, batch.size);

    auto const preprocessStart = std::chrono::steady_clock::now();
    AffineMatrices affineMatrices = preprocessBatchToGpu(model, images, batch);
    auto const preprocessEnd = std::chrono::steady_clock::now();

    inferenceTimer.start(model.stream);
    infer(model);
    std::vector<float> output = copyToCpu(model, batch.size);
    float const inferenceMilliseconds = inferenceTimer.stop("inference", false);

    auto const postprocessStart = std::chrono::steady_clock::now();
    Results results = printBatchResults(
        model,
        batch,
        affineMatrices,
        output,
        confidenceThreshold);
    auto const postprocessEnd = std::chrono::steady_clock::now();

    return {
        std::move(results),
        std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count(),
        inferenceMilliseconds,
        std::chrono::duration<double, std::milli>(postprocessEnd - postprocessStart).count()};
}

void printTiming(
    char const* itemName,
    std::size_t itemIndex,
    int batchSize,
    DetectionRun const& run)
{
    auto const previousFlags = std::cout.flags();
    auto const previousPrecision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3)
              << "timing: " << itemName << '=' << itemIndex
              << " batch=" << batchSize
              << " preprocess=" << run.preprocessMilliseconds << " ms"
              << " inference=" << run.inferenceMilliseconds << " ms"
              << " postprocess=" << run.postprocessMilliseconds << " ms"
              << " total="
              << run.preprocessMilliseconds
                    + run.inferenceMilliseconds
                    + run.postprocessMilliseconds
              << " ms\n";
    std::cout.flags(previousFlags);
    std::cout.precision(previousPrecision);
}

bool isImagePath(std::filesystem::path const& path)
{
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

    return extension == ".jpg"
        || extension == ".jpeg"
        || extension == ".png"
        || extension == ".bmp"
        || extension == ".webp"
        || extension == ".tif"
        || extension == ".tiff";
}

std::vector<std::string> findImagePaths(std::string const& imageDirectory)
{
    std::filesystem::path const directory(imageDirectory);
    Assertf(std::filesystem::is_directory(directory),
        "Image directory does not exist: %s", imageDirectory.c_str());

    std::vector<std::string> imagePaths;
    for (std::filesystem::directory_entry const& entry
        : std::filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file() && isImagePath(entry.path()))
        {
            imagePaths.push_back(entry.path().string());
        }
    }

    std::sort(imagePaths.begin(), imagePaths.end());
    Assertf(!imagePaths.empty(),
        "No supported images found in directory: %s", imageDirectory.c_str());
    return imagePaths;
}

void detectImageDirectory(
    Model& model,
    std::string const& imageDirectory,
    float confidenceThreshold)
{
    std::vector<std::string> const imagePaths = findImagePaths(imageDirectory);
    Images images = loadImages(imagePaths);
    Batches const batches = splitByMaxBatch(images.size(), model.maxBatch);

    Results results;
    results.reserve(images.size());
    trt_timer::Timer inferenceTimer;

    for (std::size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex)
    {
        Batch const& batch = batches[batchIndex];
        DetectionRun run = runDetection(
            model,
            images,
            batch,
            confidenceThreshold,
            inferenceTimer);
        printTiming("batch", batchIndex, batch.size, run);
        results.insert(results.end(), run.results.begin(), run.results.end());
    }

    showResults(images, results);
}

void detectVideo(
    Model& model,
    std::string const& videoPath,
    float confidenceThreshold)
{
    cv::VideoCapture capture(videoPath);
    Assertf(capture.isOpened(), "Cannot open video: %s", videoPath.c_str());

    std::string const windowName = "YOLO video detection";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    Batch const batch{0, 1};
    trt_timer::Timer inferenceTimer;
    std::size_t frameIndex = 0;
    double smoothedFps = 0.0;

    for (cv::Mat frame; capture.read(frame); ++frameIndex)
    {
        auto const frameStart = std::chrono::steady_clock::now();
        Images images{frame};
        DetectionRun run = runDetection(
            model,
            images,
            batch,
            confidenceThreshold,
            inferenceTimer);
        printTiming("frame", frameIndex, batch.size, run);

        cv::Mat displayFrame = drawImageResults(frame, run.results.front());
        auto const frameEnd = std::chrono::steady_clock::now();
        double const frameSeconds
            = std::chrono::duration<double>(frameEnd - frameStart).count();
        double const currentFps = frameSeconds > 0.0 ? 1.0 / frameSeconds : 0.0;
        smoothedFps = frameIndex == 0
            ? currentFps
            : smoothedFps * 0.9 + currentFps * 0.1;

        std::ostringstream fpsLabel;
        fpsLabel << "FPS: " << std::fixed << std::setprecision(1) << smoothedFps;
        cv::putText(
            displayFrame,
            fpsLabel.str(),
            cv::Point(16, 32),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 0),
            2,
            cv::LINE_AA);
        cv::imshow(windowName, displayFrame);

        int const key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }

    cv::destroyWindow(windowName);
}

} // namespace

int main()
{
    std::string const enginePath = R"(/home/autumn/CLionProjects/tensorrt/model/linux.engine)";
    std::string const videoPath = R"(/home/autumn/CLionProjects/tensorrt/model/test.mp4)";
    std::string const imageDirectory = R"(/home/autumn/CLionProjects/tensorrt/model)";
    float const confidenceThreshold = 0.25F;

    EngineData engineData = readEngine(enginePath);
    Model model = initModel(engineData);

    // detectVideo(model, videoPath, confidenceThreshold);
    detectImageDirectory(model, imageDirectory, confidenceThreshold);

    releaseModel(model);
    return 0;
}
