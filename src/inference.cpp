#include "inference.h"

#include <stdexcept>

void setBatchSize(Model& model, int batchSize)
{
    model.inputShape.d[0] = batchSize;
    model.context->setInputShape(model.inputName.c_str(), model.inputShape);
}

void copyToGpu(Model& model, cv::Mat const& input)
{
    cudaMemcpy(
        model.inputDevice,
        input.ptr<float>(),
        input.total() * sizeof(float),
        cudaMemcpyHostToDevice);
}

void infer(Model& model)
{
    if (!model.context->enqueueV3(model.stream))
    {
        throw std::runtime_error("enqueueV3 failed");
    }
    cudaStreamSynchronize(model.stream);
}

std::vector<float> copyToCpu(Model const& model, int batchSize)
{
    std::vector<float> output(
        static_cast<std::size_t>(batchSize) * model.maxDetections * 6);

    cudaMemcpy(
        output.data(),
        model.outputDevice,
        output.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    return output;
}
