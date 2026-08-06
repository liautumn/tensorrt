#pragma once

#include "batch.h"
#include "image.h"
#include "model.h"

#include <vector>

void printBatchResults(
    Model const& model,
    Images const& images,
    Batch const& batch,
    std::vector<float> const& output);
