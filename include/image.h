#pragma once

#include "batch.h"

#include <opencv2/core/mat.hpp>

#include <string>
#include <vector>

using Images = std::vector<cv::Mat>;

Images loadImages(std::vector<std::string> const& imagePaths);
cv::Mat preprocessBatch(
    Images const& images,
    Batch const& batch,
    int inputHeight,
    int inputWidth);
