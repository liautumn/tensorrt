// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

// 防止当前头文件在同一个编译单元中被重复包含。
#pragma once

// 引入 Batch，结果解析时需要知道当前批次在全部图片中的起点和数量。
#include "batch.h"
// 引入 Images；结果显示时需要把 results[i] 绘制到对应的原始图片 images[i] 上。
#include "image.h"
// 引入 Model，结果解析时需要每张图片允许的最大检测框数量。
#include "model.h"
// 引入 CUDA letterbox 生成的逆仿射矩阵类型。
#include "preprocess.h"

// 引入 std::vector，用于保存单张图片以及全部图片的检测结果。
#include <vector>

// 表示模型在一张图片中识别出的一个有效目标。
struct Detection
{
    // 目标框左上角的 x 坐标，已经通过 d2i 逆仿射矩阵映射到原图坐标系。
    float x1;
    // 目标框左上角的 y 坐标，已经通过 d2i 逆仿射矩阵映射到原图坐标系。
    float y1;
    // 目标框右下角的 x 坐标，已经通过 d2i 逆仿射矩阵映射到原图坐标系。
    float x2;
    // 目标框右下角的 y 坐标，已经通过 d2i 逆仿射矩阵映射到原图坐标系。
    float y2;
    // 模型给出的目标置信度，用于判断该检测框是否可信。
    float confidence;
    // 模型给出的类别编号，例如具体编号对应的人、车等类别由模型标签定义。
    int classId;
};

// 一张图片的检测结果：vector 中的每个 Detection 对应一个有效目标。
using ImageResults = std::vector<Detection>;
// 一个批次的检测结果：外层 vector 的下标与当前批次内的图片下标一一对应。
using Results = std::vector<ImageResults>;

// 解析一个批次的模型输出，只保留达到置信度阈值的检测框。
// model：模型信息，提供每张图片允许的最大检测框数量。
// batch：当前批次信息，offset 表示起始图片下标，size 表示本批图片数量。
// affineMatrices：当前批次各图片的网络坐标到原图坐标 d2i，顺序必须与 batch 一致。
// output：从 GPU 复制回来的本批次输出，布局为 [batchSize, maxDetections, 6]。
// confidenceThreshold：最低置信度；小于该值的检测框不会放入返回集合。
// 返回值：当前批次的有效结果集合；即使某张图片没有有效框，也保留对应的空集合。
Results printBatchResults(
    Model const& model,
    Batch const& batch,
    AffineMatrices const& affineMatrices,
    std::vector<float> const& output,
    float confidenceThreshold);

// 在原图副本上绘制一张图片的全部检测框和标签。
// image：待绘制的原始 BGR 图片，函数不会修改它。
// results：与 image 对应的检测结果。
// 返回值：已经画好检测框和标签的独立图片，可直接交给 cv::imshow()。
cv::Mat drawImageResults(cv::Mat const& image, ImageResults const& results);

// 使用已经汇总完成的 results 绘制并显示全部推理结果。
// images[i] 与 results[i] 必须一一对应；函数会在原图副本上画框，不会修改原始图片或检测数据。
// 每张图片使用一个包含全局下标的独立 OpenCV 窗口，并显示类别编号和置信度。
// 所有窗口创建完成后，函数等待用户在任意结果窗口中按键，然后统一关闭窗口。
void showResults(Images const& images, Results const& results);
