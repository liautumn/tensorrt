#pragma once

#include "batch.h"
#include "image.h"
#include "model.h"

#include <vector>

struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int classId;
};

using ImageResults = std::vector<Detection>;
using Results = std::vector<ImageResults>;

Results printBatchResults(
    Model const& model,
    Images const& images,
    Batch const& batch,
    std::vector<float> const& output,
    float confidenceThreshold);
