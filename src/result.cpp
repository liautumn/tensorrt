// 引入 Detection、Results 和 printBatchResults 的声明，以及相关数据类型。
#include "result.h"

// 引入 std::cout，用于把每张图片的有效检测结果打印到控制台。
#include <iostream>

// 解析当前批次的一维模型输出，打印有效框，并返回结构化结果集合。
// model 提供模型输入尺寸和 maxDetections；images 提供原图尺寸。
// batch 指明当前批次对应全部图片中的哪一段；output 是当前批次的 CPU 输出。
// confidenceThreshold 决定哪些候选框属于有效结果。
Results printBatchResults(
    // 只读取模型元数据，不修改模型，因此使用常量引用。
    Model const& model,
    // 全部原始图片，使用 batch.offset 定位到本轮图片，不复制图像数据。
    Images const& images,
    // 当前批次的起始下标 offset 和实际图片数量 size。
    Batch const& batch,
    // copyToCpu 返回的一维 float 输出，使用常量引用避免复制整份结果。
    std::vector<float> const& output,
    // 检测框的最低置信度，只保留 confidence 大于或等于该值的框。
    float confidenceThreshold)
{
    // 为当前批次的每张图片创建一个结果集合。
    // 外层长度固定为 batch.size，没有检测框的图片会对应一个空 vector。
    Results results(batch.size);

    // b 是图片在“当前批次”中的下标，范围从 0 到 batch.size - 1。
    for (int b = 0; b < batch.size; ++b)
    {
        // batch.offset 是当前批次在全部图片中的起点，加 b 得到图片的全局下标。
        // 使用常量引用，避免复制 cv::Mat 对象。
        cv::Mat const& image = images[batch.offset + b];
        // 计算模型输入宽度到原图宽度的缩放比例，用于还原 x 坐标。
        float const scaleX = static_cast<float>(image.cols) / model.inputWidth;
        // 计算模型输入高度到原图高度的缩放比例，用于还原 y 坐标。
        float const scaleY = static_cast<float>(image.rows) / model.inputHeight;

        // 打印当前图片在全部输入图片中的下标，前面的换行用于分隔不同图片。
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
                Detection detection{
                    // item[0] 是模型输入坐标系中的 x1，乘宽度比例还原到原图。
                    item[0] * scaleX,
                    // item[1] 是模型输入坐标系中的 y1，乘高度比例还原到原图。
                    item[1] * scaleY,
                    // item[2] 是模型输入坐标系中的 x2，乘宽度比例还原到原图。
                    item[2] * scaleX,
                    // item[3] 是模型输入坐标系中的 y2，乘高度比例还原到原图。
                    item[3] * scaleY,
                    // item[4] 直接保存模型给出的置信度。
                    item[4],
                    // item[5] 在输出中是 float，这里转换为 Detection 使用的整数类别编号。
                    static_cast<int>(item[5])};

                // results[b] 对应当前批次第 b 张图片，把有效检测框加入它的结果集合。
                results[b].push_back(detection);
                // 将类别、置信度和原图坐标打印到控制台，便于直接观察推理结果。
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
