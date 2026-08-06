#pragma once

#include "model.h"

#include <opencv2/core/mat.hpp>

#include <vector>

void setBatchSize(Model& model, int batchSize);
void copyToGpu(Model& model, cv::Mat const& input);
void infer(Model& model);
std::vector<float> copyToCpu(Model const& model, int batchSize);
