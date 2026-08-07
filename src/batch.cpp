// 引入 Batch、Batches 和 splitByMaxBatch 的声明。
#include "batch.h"

// 提供 std::min，用来决定当前批次实际取多少张图片。
#include <algorithm>

// 使用 Assertf 拒绝会让分批循环无法推进的最大 batch。
#include "validator.h"

// 将 imageCount 张图片按 maxBatch 从前到后切分，图片顺序不会改变。
Batches splitByMaxBatch(std::size_t imageCount, int maxBatch)
{
    // maxBatch 为 0 或负数时 offset 无法可靠向后推进。
    Assertf(maxBatch > 0, "Maximum batch size must be positive, got %d", maxBatch);
    // 创建空集合，用来依次保存每一轮推理的批次信息。
    Batches batches;

    // offset 是当前批次第一张图片的下标；只要还有图片没有分批，就继续循环。
    for (std::size_t offset = 0; offset < imageCount;)
    {
        // 当前批次取“模型最大 batch”和“剩余图片数”中的较小值。
        // 例如共有 10 张图片、maxBatch 为 4，三次得到的 batchSize 是 4、4、2。
        int const batchSize
            = std::min<int>(maxBatch, static_cast<int>(imageCount - offset));

        // 保存当前批次：从 offset 开始，共包含 batchSize 张图片。
        batches.push_back({offset, batchSize});

        // 跳过刚刚分入当前批次的图片，让 offset 指向下一批的第一张图片。
        offset += batchSize;
    }

    // 返回全部批次；后续代码会按这个顺序逐批预处理和推理。
    return batches;
}
