#include "image.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

#include <stdexcept>

Images loadImages(std::vector<std::string> const& imagePaths)
{
    Images images;
    for (auto const& path : imagePaths)
    {
        cv::Mat image = cv::imread(path);
        if (image.empty())
        {
            throw std::runtime_error("Cannot open image: " + path);
        }
        images.push_back(image);
    }
    return images;
}

cv::Mat preprocessBatch(
    Images const& images,
    Batch const& batch,
    int inputHeight,
    int inputWidth)
{
    Images batchImages;
    for (int i = 0; i < batch.size; ++i)
    {
        batchImages.push_back(images[batch.offset + i]);
    }

    cv::Mat input;
    cv::dnn::blobFromImages(
        batchImages,
        input,
        1.0 / 255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(),
        true,
        false,
        CV_32F);
    return input;
}
