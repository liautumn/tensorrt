// 防止同一个头文件在一次编译过程中被重复包含。
#pragma once

// 引入 Batch，preprocessBatch 需要用它确定当前处理哪一段图片。
#include "batch.h"

// 引入 OpenCV 的 cv::Mat；图片和预处理后的模型输入都用它保存。
#include <opencv2/core/mat.hpp>

// 提供 std::string，用来保存图片文件路径。
#include <string>
// 提供 std::vector，用来保存多条路径或多张图片。
#include <vector>

// Images 是 std::vector<cv::Mat> 的简短别名，表示按输入顺序保存的多张图片。
using Images = std::vector<cv::Mat>;

// 按顺序读取多张图片。
// 参数 imagePaths：图片路径集合，以 const 引用传入，函数只读取而不复制、不修改它。
// 返回值：读取成功的图片集合；返回图片的顺序与 imagePaths 完全一致。
// 异常：任意图片读取失败时抛出 std::runtime_error，不返回不完整的结果。
Images loadImages(std::vector<std::string> const& imagePaths);

// 取出一个批次的图片，并转换成模型可以接收的浮点输入数据。
// 参数 images：已经读取好的全部图片，函数只读取它们。
// 参数 batch：本轮要处理的起始下标 offset 和实际图片数 size。
// 参数 inputHeight：模型要求的单张输入图片高度。
// 参数 inputWidth：模型要求的单张输入图片宽度。
// 返回值：连续的 CV_32F 数据，排列形式为 [batch, channel, height, width]。
cv::Mat preprocessBatch(
    // 全部原始图片。
    Images const& images,
    // 当前批次在全部图片中的位置和数量。
    Batch const& batch,
    // 模型输入高度。
    int inputHeight,
    // 模型输入宽度。
    int inputWidth);
