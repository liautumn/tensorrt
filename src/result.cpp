// 引入 Detection、Results 和 printBatchResults 的声明，以及相关数据类型。
#include "result.h"

// 取消下方调试输出的注释时，std::cout 用于打印每张图片的有效检测结果。
#include <iostream>
// std::invalid_argument 用于报告仿射矩阵数量与 batch 不匹配。
#include <stdexcept>

// 解析当前批次的一维模型输出，并返回结构化结果集合。
// model 提供 maxDetections；batch 指明当前批次对应全部图片中的哪一段。
// affineMatrices 提供每张图的 d2i；output 是当前批次的 CPU 输出。
// confidenceThreshold 决定哪些候选框属于有效结果。
Results printBatchResults(
    // 只读取模型元数据，不修改模型，因此使用常量引用。
    Model const& model,
    // 当前批次的起始下标 offset 和实际图片数量 size。
    Batch const& batch,
    // CUDA letterbox 为当前批次每张图片生成的网络坐标到原图坐标逆矩阵。
    AffineMatrices const& affineMatrices,
    // copyToCpu 返回的一维 float 输出，使用常量引用避免复制整份结果。
    std::vector<float> const& output,
    // 检测框的最低置信度，只保留 confidence 大于或等于该值的框。
    float confidenceThreshold)
{
    // 每张图片必须恰好对应一组 d2i；否则检测框可能使用另一张图的逆变换。
    if (affineMatrices.size() != static_cast<std::size_t>(batch.size))
    {
        throw std::invalid_argument("affine matrix count does not match batch size");
    }

    // 为当前批次的每张图片创建一个结果集合。
    // 外层长度固定为 batch.size，没有检测框的图片会对应一个空 vector。
    Results results(batch.size);

    // b 是图片在“当前批次”中的下标，范围从 0 到 batch.size - 1。
    for (int b = 0; b < batch.size; ++b)
    {
        // affineMatrices 按批次内下标排列；d2i 将网络坐标映射回这一张原图。
        std::array<float, 6> const& d2i = affineMatrices[b].d2i;

        // 可选调试输出：打印当前图片在全部输入图片中的下标。
        std::cout << "\nimage " << batch.offset + b << '\n';
        // 逐个检查模型为当前图片预留的所有候选检测框。
        for (int i = 0; i < model.maxDetections; ++i)
        {
            // item 指向第 b 张图片的第 i 个候选框在一维 output 中的起始位置。
            // b * maxDetections 跳过前面图片的框，加 i 定位当前框，再乘 6 定位其字段。
            float const* item
                = output.data() + (b * model.maxDetections + i) * 6;
            // item[4] 是置信度；仅处理达到调用方传入阈值的候选框。
            if (item[4] >= confidenceThreshold)
            {
                // 把模型的 6 个 float 字段转换为更容易使用的 Detection 结构体。
                // 对左上角 (x1,y1) 应用 d2i 第一行，得到原图 x1。
                float const projectedX1 = d2i[0] * item[0] + d2i[1] * item[1] + d2i[2];
                // 对同一个左上角应用 d2i 第二行，得到原图 y1。
                float const projectedY1 = d2i[3] * item[0] + d2i[4] * item[1] + d2i[5];
                // 右下角 (x2,y2) 必须独立应用第一行，不能再使用旧的 scaleX。
                float const projectedX2 = d2i[0] * item[2] + d2i[1] * item[3] + d2i[2];
                // 对右下角应用第二行，得到原图 y2。
                float const projectedY2 = d2i[3] * item[2] + d2i[4] * item[3] + d2i[5];

                Detection detection{
                    // 与目标分支一致，保留映射后的 x1，不在这里额外裁剪图片边界。
                    projectedX1,
                    // 映射后的原图 y1。
                    projectedY1,
                    // 映射后的原图 x2。
                    projectedX2,
                    // 映射后的原图 y2。
                    projectedY2,
                    // item[4] 直接保存模型给出的置信度。
                    item[4],
                    // item[5] 在输出中是 float，这里转换为 Detection 使用的整数类别编号。
                    static_cast<int>(item[5])};

                // results[b] 对应当前批次第 b 张图片，把有效检测框加入它的结果集合。
                results[b].push_back(detection);
                // 可选调试输出：打印类别、置信度和还原后的原图坐标。
                std::cout << "class=" << detection.classId
                          << " score=" << detection.confidence
                          << " box=[" << detection.x1
                          << ',' << detection.y1
                          << ',' << detection.x2
                          << ',' << detection.y2 << "]\n";
            }
        }
    }
    // 返回当前批次的结构化结果，外层下标仍是批次内图片下标。
    return results;
}
