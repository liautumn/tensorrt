#pragma once

#include <cstddef>
#include <vector>

struct Batch
{
    std::size_t offset;
    int size;
};

using Batches = std::vector<Batch>;

Batches splitByMaxBatch(std::size_t imageCount, int maxBatch);
