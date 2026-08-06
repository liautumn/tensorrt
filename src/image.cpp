// 引入 Images、Batch、loadImages 和 preprocessBatch 的声明。
#include "image.h"

// 提供 cv::dnn::blobFromImages，用来把多张图片转换成模型输入张量。
#include <opencv2/dnn.hpp>
// 提供 cv::imread，用来从磁盘读取图片。
#include <opencv2/imgcodecs.hpp>

// 提供 std::runtime_error，图片读取失败时用它报告错误。
#include <stdexcept>

// 读取 imagePaths 中列出的全部图片，并保持原有顺序。
Images loadImages(std::vector<std::string> const& imagePaths)
{
    // 创建空图片集合，用来保存读取成功的 cv::Mat。
    Images images;

    // 依次访问每一条路径；const 引用避免复制字符串，也不会修改路径。
    for (auto const& path : imagePaths)
    {
        // 使用 OpenCV 默认方式读取图片；彩色图片的通道顺序默认是 BGR。
        cv::Mat image = cv::imread(path);

        // empty() 为 true 表示文件不存在、格式不支持或图片数据无法读取。
        if (image.empty())
        {
            // 立即停止并在异常信息中带上失败的路径，避免把无效图片送入模型。
            throw std::runtime_error("Cannot open image: " + path);
        }

        // 把有效图片追加到集合末尾，因此结果顺序与传入路径顺序一致。
        images.push_back(image);
    }

    // 返回全部读取成功的图片。
    return images;
}

// 从全部图片中取出 batch 指定的一段，并完成模型输入预处理。
cv::Mat preprocessBatch(
    // images 保存本次任务的全部原始图片，函数不会修改它们。
    Images const& images,
    // batch.offset 指定起点，batch.size 指定本轮图片数量。
    Batch const& batch,
    // inputHeight 是模型要求的输入高度。
    int inputHeight,
    // inputWidth 是模型要求的输入宽度。
    int inputWidth)
{
    // 创建当前批次的临时图片集合，只保存本轮需要处理的图片。
    Images batchImages;

    // 从 0 循环到 batch.size - 1，把当前批次的每张图片依次取出。
    for (int i = 0; i < batch.size; ++i)
    {
        // batch.offset + i 是这张图片在全部 images 中的下标。
        batchImages.push_back(images[batch.offset + i]);
    }

    // 创建输出 cv::Mat；blobFromImages 会在这里写入预处理后的连续浮点数据。
    cv::Mat input;

    // 一次处理当前批次中的全部图片，并生成通常为 [N,C,H,W] 的模型输入。
    cv::dnn::blobFromImages(
        // 输入：当前批次的原始 BGR 图片。
        batchImages,
        // 输出：预处理后的模型输入数据。
        input,
        // 缩放系数：每个像素值乘以 1/255，将常见的 0~255 转为 0~1。
        1.0 / 255.0,
        // 目标尺寸：cv::Size 的参数顺序是宽、高，因此这里先传 inputWidth。
        cv::Size(inputWidth, inputHeight),
        // 均值：空 Scalar 等于各通道都减 0，本代码不做均值归一化。
        cv::Scalar(),
        // swapRB=true：把 OpenCV 的 BGR 通道顺序转换为模型常用的 RGB。
        true,
        // crop=false：缩放到目标尺寸后不再进行中心裁剪。
        false,
        // 输出类型：每个输入元素都使用 32 位浮点数保存。
        CV_32F);

    // 返回预处理结果，后续会把这块数据复制到 GPU 输入显存。
    return input;
}
