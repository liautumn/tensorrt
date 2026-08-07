// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

// 防止同一个头文件在一次编译过程中被重复包含。
#pragma once

// 提供 std::size_t，用来表示不会为负的图片下标和图片数量。
#include <cstddef>
// 提供 std::vector，用来保存全部批次信息。
#include <vector>

// Batch 描述“一轮推理应该处理原图片集合中的哪一段”。
struct Batch
{
    // 当前批次第一张图片在原图片集合中的下标，从 0 开始。
    std::size_t offset;
    // 当前批次实际包含的图片数量；最后一批可以小于模型的最大 batch。
    int size;
};

// Batches 是 std::vector<Batch> 的简短别名，表示按顺序排列的全部批次。
using Batches = std::vector<Batch>;

// 按模型允许的最大 batch 数量，把 imageCount 张图片划分成多个连续批次。
// 参数 imageCount：本次一共需要推理多少张图片。
// 参数 maxBatch：模型一次推理最多允许传入多少张图片，调用时必须大于 0。
// 返回值：全部批次的 offset 和 size；imageCount 为 0 时返回空集合。
Batches splitByMaxBatch(std::size_t imageCount, int maxBatch);
