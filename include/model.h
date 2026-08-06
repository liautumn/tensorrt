#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>

using EngineData = std::vector<char>;

struct Model
{
    nvinfer1::IRuntime* runtime{};
    nvinfer1::ICudaEngine* engine{};
    nvinfer1::IExecutionContext* context{};
    cudaStream_t stream{};
    void* inputDevice{};
    void* outputDevice{};

    std::string inputName{"images"};
    std::string outputName{"output0"};
    nvinfer1::Dims inputShape{};
    int inputHeight{};
    int inputWidth{};
    int maxBatch{};
    int maxDetections{};
};

EngineData readEngine(std::string const& enginePath);
Model initModel(EngineData const& engineData);
void releaseModel(Model& model);
