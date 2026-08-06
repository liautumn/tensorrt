#include "batch.h"

#include <algorithm>

Batches splitByMaxBatch(std::size_t imageCount, int maxBatch)
{
    Batches batches;
    for (std::size_t offset = 0; offset < imageCount;)
    {
        int const batchSize
            = std::min<int>(maxBatch, static_cast<int>(imageCount - offset));
        batches.push_back({offset, batchSize});
        offset += batchSize;
    }
    return batches;
}
