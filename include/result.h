// 防止当前头文件在同一个编译单元中被重复包含。
#pragma once

// 引入 Batch，结果解析时需要知道当前批次在全部图片中的起点和数量。
#include "batch.h"
// 引入 Images，结果坐标需要根据每张原图的宽高进行缩放。
#include "image.h"
// 引入 Model，结果解析时需要模型输入尺寸和最大检测框数量。
#include "model.h"

// 引入 std::vector，用于保存单张图片以及全部图片的检测结果。
#include <vector>

// 表示模型在一张图片中识别出的一个有效目标。
struct Detection
{
    // 目标框左上角的 x 坐标，已经从模型输入尺寸缩放到原图尺寸。
    float x1;
    // 目标框左上角的 y 坐标，已经从模型输入尺寸缩放到原图尺寸。
    float y1;
    // 目标框右下角的 x 坐标，已经从模型输入尺寸缩放到原图尺寸。
    float x2;
    // 目标框右下角的 y 坐标，已经从模型输入尺寸缩放到原图尺寸。
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

// 解析并打印一个批次的模型输出，只保留达到置信度阈值的检测框。
// model：模型信息，提供输入宽高和每张图片允许的最大检测框数量。
// images：全部原始图片，用于取得当前批次中每张图片的原始尺寸。
// batch：当前批次信息，offset 表示起始图片下标，size 表示本批图片数量。
// output：从 GPU 复制回来的本批次输出，布局为 [batchSize, maxDetections, 6]。
// confidenceThreshold：最低置信度；小于该值的检测框不会打印，也不会放入返回集合。
// 返回值：当前批次的有效结果集合；即使某张图片没有有效框，也保留对应的空集合。
Results printBatchResults(
    Model const& model,
    Images const& images,
    Batch const& batch,
    std::vector<float> const& output,
    float confidenceThreshold);
