#include "model.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace
{

struct Logger : nvinfer1::ILogger
{
    void log(Severity severity, char const* message) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            fprintf(stderr, "%s\n", message);
        }
    }
};

Logger logger;

} // namespace

EngineData readEngine(std::string const& enginePath)
{
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Cannot open engine: " + enginePath);
    }

    EngineData engineData(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    file.read(engineData.data(), static_cast<std::streamsize>(engineData.size()));
    return engineData;
}

Model initModel(EngineData const& engineData)
{
    Model model;
    model.runtime = nvinfer1::createInferRuntime(logger);
    model.engine = model.runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    model.context = model.engine->createExecutionContext();

    auto const optShape = model.engine->getProfileShape(
        model.inputName.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
    auto const maxShape = model.engine->getProfileShape(
        model.inputName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);

    model.inputShape = optShape;
    model.inputHeight = static_cast<int>(optShape.d[2]);
    model.inputWidth = static_cast<int>(optShape.d[3]);
    model.maxBatch = static_cast<int>(maxShape.d[0]);

    cudaStreamCreate(&model.stream);
    model.context->setOptimizationProfileAsync(0, model.stream);

    model.inputShape.d[0] = model.maxBatch;
    model.context->setInputShape(model.inputName.c_str(), model.inputShape);
    model.maxDetections
        = static_cast<int>(model.context->getTensorShape(model.outputName.c_str()).d[1]);

    cudaMalloc(&model.inputDevice,
        static_cast<std::size_t>(model.maxBatch) * 3 * model.inputHeight * model.inputWidth * sizeof(float));
    cudaMalloc(&model.outputDevice,
        static_cast<std::size_t>(model.maxBatch) * model.maxDetections * 6 * sizeof(float));

    model.context->setTensorAddress(model.inputName.c_str(), model.inputDevice);
    model.context->setTensorAddress(model.outputName.c_str(), model.outputDevice);
    return model;
}

void releaseModel(Model& model)
{
    cudaFree(model.inputDevice);
    cudaFree(model.outputDevice);
    cudaStreamDestroy(model.stream);
    delete model.context;
    delete model.engine;
    delete model.runtime;
}
