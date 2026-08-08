// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

// 引入 Detection、Results 和 printBatchResults 的声明，以及相关数据类型。
#include "result.h"

// highgui 提供 namedWindow()、imshow()、waitKey() 和 destroyAllWindows()。
#include <opencv2/highgui.hpp>
// imgproc 提供 rectangle()、putText() 和 getTextSize()。
#include <opencv2/imgproc.hpp>

// std::clamp 用于把仅供显示的检测框坐标限制在图片边界内。
#include <algorithm>
// std::isfinite 和 std::lround 用于检查浮点坐标并将其转换为像素坐标。
#include <cmath>
// std::fixed 和 std::setprecision() 用于把标签置信度固定显示为两位小数。
#include <iomanip>
// std::cout 用于窗口等待提示；检测结果改由 Validator 同时输出到控制台和日志。
#include <iostream>
// std::numeric_limits 用于确认模型给出的浮点类别编号可以安全转换为 int。
#include <limits>
// std::ostringstream 用于拼接类别编号和置信度标签。
#include <sstream>
// std::string 和 std::to_string() 用于生成标签及每张图片的唯一窗口名称。
#include <string>

// 使用统一断言检查集合尺寸契约，并把图片及检测结果写入每日信息日志。
#include "validator.h"

namespace
{
// 在一张可显示的图片副本上绘制一个 Detection。
// Detection 中保存的是未裁边的原图浮点坐标；这里的裁边和取整只服务于可视化，
// 不会反向修改 results，调用方仍可读取模型后处理得到的原始坐标。
void drawDetection(cv::Mat& displayImage, Detection const& detection)
{
    // 非有限坐标无法安全转换成整数像素；遇到 NaN 或无穷大时直接跳过该框。
    if (!std::isfinite(detection.x1)
        || !std::isfinite(detection.y1)
        || !std::isfinite(detection.x2)
        || !std::isfinite(detection.y2)
        || !std::isfinite(detection.confidence))
    {
        return;
    }

    // OpenCV 有效像素下标分别截止到 cols - 1 和 rows - 1。
    float const maxX = static_cast<float>(displayImage.cols - 1);
    float const maxY = static_cast<float>(displayImage.rows - 1);

    // d2i 还原后的框可能略微超出原图，所以绘制前分别限制四条边。
    float const clippedX1 = std::clamp(detection.x1, 0.0F, maxX);
    float const clippedY1 = std::clamp(detection.y1, 0.0F, maxY);
    float const clippedX2 = std::clamp(detection.x2, 0.0F, maxX);
    float const clippedY2 = std::clamp(detection.y2, 0.0F, maxY);

    // lround 按最近像素取整；坐标已经裁边，因此转换后的整数一定落在图片范围内。
    int const left = static_cast<int>(std::lround(clippedX1));
    int const top = static_cast<int>(std::lround(clippedY1));
    int const right = static_cast<int>(std::lround(clippedX2));
    int const bottom = static_cast<int>(std::lround(clippedY2));

    // 裁边或取整后宽高为 0 的框没有可见面积，也不能构造有效的显示区域。
    if (right <= left || bottom <= top)
    {
        return;
    }

    // OpenCV 的 Scalar 按 BGR 排列；这里使用绿色同时绘制边框和标签背景。
    cv::Scalar const boxColor(0, 255, 0);
    // LINE_AA 让缩放窗口时的框线边缘更平滑，2 表示边框宽度为两个像素。
    cv::rectangle(
        displayImage,
        cv::Point(left, top),
        cv::Point(right, bottom),
        boxColor,
        2,
        cv::LINE_AA);

    // 标签直接从当前 Detection 读取，不再访问 TensorRT 的原始 output。
    std::ostringstream labelStream;
    labelStream << "class=" << detection.classId
                << " score=" << std::fixed << std::setprecision(2) << detection.confidence;
    std::string const label = labelStream.str();

    // 先测量文字尺寸，随后为文字计算一个不会超出图片边界的实色背景区域。
    int constexpr fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double constexpr fontScale = 0.6;
    int constexpr fontThickness = 1;
    int constexpr padding = 4;
    int baseline = 0;
    cv::Size const textSize = cv::getTextSize(
        label,
        fontFace,
        fontScale,
        fontThickness,
        &baseline);

    // 极小图片可能容不下完整标签，所以背景宽高最多取图片自身宽高。
    int const labelWidth = std::min(textSize.width + padding * 2, displayImage.cols);
    int const labelHeight
        = std::min(textSize.height + baseline + padding * 2, displayImage.rows);
    // 标签横向优先与检测框左边对齐；太靠右时整体左移到图片内。
    int const labelLeft = std::clamp(left, 0, displayImage.cols - labelWidth);
    // 空间足够时把标签放在框上方，否则放在框顶边附近，并再次限制到图片内。
    int const preferredLabelTop = top >= labelHeight ? top - labelHeight : top + 1;
    int const labelTop = std::clamp(
        preferredLabelTop,
        0,
        displayImage.rows - labelHeight);

    // 实色背景避免类别和置信度文字被复杂的原图纹理淹没。
    cv::rectangle(
        displayImage,
        cv::Rect(labelLeft, labelTop, labelWidth, labelHeight),
        boxColor,
        cv::FILLED);

    // 文字原点表示基线位置；即使图片极小，min() 也能保证原点仍位于图片范围内。
    int const textX = std::min(labelLeft + padding, displayImage.cols - 1);
    int const textY
        = std::min(labelTop + padding + textSize.height, displayImage.rows - 1);
    // 黑色文字与绿色背景形成稳定对比，标签内容为 class 和 score。
    cv::putText(
        displayImage,
        label,
        cv::Point(textX, textY),
        fontFace,
        fontScale,
        cv::Scalar(0, 0, 0),
        fontThickness,
        cv::LINE_AA);
}
} // namespace

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
    Assertf(affineMatrices.size() == static_cast<std::size_t>(batch.size),
        "Affine matrix count does not match batch size");
    // 防止后续按 [batch,max_det,6] 取指针时越过 output 的有效范围。
    Assertf(output.size() == static_cast<std::size_t>(batch.size) * model.maxDetections * 6,
        "Output element count %zu does not match batch shape", output.size());

    // 为当前批次的每张图片创建一个结果集合。
    // 外层长度固定为 batch.size，没有检测框的图片会对应一个空 vector。
    Results results(batch.size);

    // b 是图片在“当前批次”中的下标，范围从 0 到 batch.size - 1。
    for (int b = 0; b < batch.size; ++b)
    {
        // affineMatrices 按批次内下标排列；d2i 将网络坐标映射回这一张原图。
        std::array<float, 6> const& d2i = affineMatrices[b].d2i;

        // 记录当前图片在全部输入图片中的下标，同时输出到控制台和每日文件。
        // Validator::info("image " + std::to_string(batch.offset + b));
        // 逐个检查模型为当前图片预留的所有候选检测框。
        for (int i = 0; i < model.maxDetections; ++i)
        {
            // item 指向第 b 张图片的第 i 个候选框在一维 output 中的起始位置。
            // 使用 size_t 计算偏移，避免较大 batch 和 maxDetections 的 int 乘法溢出。
            float const* item
                = output.data()
                + (static_cast<std::size_t>(b) * model.maxDetections + i) * 6;
            // item[4] 是置信度；仅处理达到调用方传入阈值的候选框。
            if (item[4] >= confidenceThreshold)
            {
                // 非有限或超出 int 范围的类别编号不能执行浮点到整数转换。
                Assertf(std::isfinite(item[5])
                        && static_cast<double>(item[5]) >= std::numeric_limits<int>::min()
                        && static_cast<double>(item[5]) <= std::numeric_limits<int>::max(),
                    "Invalid class id %g at batch item %d, detection %d", item[5], b, i);
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
                // 保持原有字段顺序，把类别、置信度和还原坐标同时写到控制台及日志。
                // std::ostringstream detectionMessage;
                // detectionMessage << "class=" << detection.classId
                //                  << " score=" << detection.confidence
                //                  << " box=[" << detection.x1
                //                  << ',' << detection.y1
                //                  << ',' << detection.x2
                //                  << ',' << detection.y2 << ']';
                // Validator::info(detectionMessage.str());
            }
        }
    }
    // 返回当前批次的结构化结果，外层下标仍是批次内图片下标。
    return results;
}

// 从汇总后的 results 获取检测数据，在原图副本上绘制并通过 OpenCV 窗口显示。
void showResults(Images const& images, Results const& results)
{
    // main() 按批次顺序把 batchResults 追加到 results，因此两者正常情况下长度相同。
    // 若长度不一致，继续按下标访问会导致图片与结果错配，必须立即报告错误。
    Assertf(images.size() == results.size(), "Image count does not match result count");

    // 没有输入图片时不创建窗口，也不能调用无限等待的 waitKey(0)。
    if (images.empty())
    {
        return;
    }

    // results[i] 对应 images[i]；这里使用全局下标，而不是任意一个批次内的局部下标。
    for (std::size_t imageIndex = 0; imageIndex < images.size(); ++imageIndex)
    {
        // loadImages() 已保证图片有效；这里再次防守，避免对空 Mat 计算 cols - 1。
        Assertf(!images[imageIndex].empty(), "Cannot display an empty image");

        // clone() 创建独立像素缓冲区，画框不会污染 images 中保存的原始 BGR 图片。
        cv::Mat displayImage = images[imageIndex].clone();
        // 当前图片的所有框都直接来自 results[imageIndex]。
        for (Detection const& detection : results[imageIndex])
        {
            drawDetection(displayImage, detection);
        }

        // 全局图片下标保证多批次、多图片情况下每个窗口名称都唯一。
        std::string const windowName = "result " + std::to_string(imageIndex);
        // WINDOW_NORMAL 允许用户调整窗口大小，OpenCV 会同步缩放显示内容。
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        // imshow() 只提交待显示图片；后面的 waitKey() 负责处理窗口刷新和键盘事件。
        cv::imshow(windowName, displayImage);
    }

    // 所有推理与计时都已结束后才进入这里，因此等待键盘不会计入任何 timing 字段。
    std::cout << "\nPress any key in an OpenCV result window to close all windows.\n";
    // 0 表示一直等待；用户在任意结果窗口按键后返回。
    cv::waitKey(0);
    // 统一销毁本函数创建的全部结果窗口，避免窗口资源一直保留到进程退出。
    cv::destroyAllWindows();
}
