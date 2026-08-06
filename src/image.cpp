// 引入 Images 和 loadImages 的声明。
#include "image.h"

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
