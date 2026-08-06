#include "result.h"

#include <iostream>

void printBatchResults(
    Model const& model,
    Images const& images,
    Batch const& batch,
    std::vector<float> const& output)
{
    for (int b = 0; b < batch.size; ++b)
    {
        cv::Mat const& image = images[batch.offset + b];
        float const scaleX = static_cast<float>(image.cols) / model.inputWidth;
        float const scaleY = static_cast<float>(image.rows) / model.inputHeight;

        std::cout << "\nimage " << batch.offset + b << '\n';
        for (int i = 0; i < model.maxDetections; ++i)
        {
            float const* item
                = output.data() + (b * model.maxDetections + i) * 6;
            if (item[4] >= 0.25F)
            {
                std::cout << "class=" << static_cast<int>(item[5])
                          << " score=" << item[4]
                          << " box=[" << item[0] * scaleX
                          << ',' << item[1] * scaleY
                          << ',' << item[2] * scaleX
                          << ',' << item[3] * scaleY << "]\n";
            }
        }
    }
}
