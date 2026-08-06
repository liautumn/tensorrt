// 防止 CUDA 预处理接口在同一编译单元中被重复包含。
#pragma once

// 引入 Batch，预处理函数用它定位当前需要处理的图片区间。
#include "batch.h"
// 引入 Images；其中每个 cv::Mat 都是待上传的原始 BGR 图片。
#include "image.h"
// 引入 Model；预处理函数通过它访问输入显存、CUDA stream 和可复用 workspace。
#include "model.h"

// std::array 保存固定长度的 2x3 仿射矩阵。
#include <array>
// std::vector 保存当前批次每张图片对应的仿射矩阵。
#include <vector>

// 保存一张图片在原图坐标系与模型输入坐标系之间的双向 2x3 仿射矩阵。
struct AffineMatrix
{
    // image-to-destination：按行保存，把原图像素坐标映射到 letterbox 后的模型输入坐标。
    std::array<float, 6> i2d{};
    // destination-to-image：i2d 的逆矩阵，供 CUDA 反采样和检测框坐标还原共用。
    std::array<float, 6> d2i{};
};

// 当前批次每张图片各自拥有一组仿射矩阵，顺序与批次内图片顺序一致。
using AffineMatrices = std::vector<AffineMatrix>;

// 使用 pinned memory、异步 H2D 和 CUDA kernel 完成当前批次的 letterbox 预处理。
// kernel 直接把 FP32 RGB NCHW 结果写入 model.inputDevice；返回值供后处理还原坐标。
// model：必须已经由 initModel() 初始化；函数会懒分配并复用其中的预处理 workspace。
// images：全部已读取的原始图片；每张待处理图片必须是非空 CV_8UC3。
// batch：指定本轮的起始下标和实际图片数，size 不能超过 Engine 的 max batch。
// 返回值：与当前 batch 等长，元素顺序和 batch 内图片顺序完全一致。
// 时序：H2D 使用 cudaMemcpyAsync 提交，但函数返回前会同步 model.stream。
AffineMatrices preprocessBatchToGpu(
    Model& model,
    Images const& images,
    Batch const& batch);
