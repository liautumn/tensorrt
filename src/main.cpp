#include "batch.h"
#include "image.h"
#include "inference.h"
#include "model.h"
#include "result.h"

#include <string>
#include <vector>

int main()
{
    std::string const enginePath = "best.engine";
    std::vector<std::string> const imagePaths{"1.jpg", "2.jpg"};
    float const confidenceThreshold = 0.25F;

    EngineData engineData = readEngine(enginePath);
    Model model = initModel(engineData);
    Images images = loadImages(imagePaths);
    Batches batches = splitByMaxBatch(images.size(), model.maxBatch);
    Results results;
    results.reserve(images.size());

    for (Batch const& batch : batches)
    {
        cv::Mat input = preprocessBatch(
            images,
            batch,
            model.inputHeight,
            model.inputWidth);

        setBatchSize(model, batch.size);
        copyToGpu(model, input);
        infer(model);
        std::vector<float> output = copyToCpu(model, batch.size);
        Results batchResults = printBatchResults(
            model,
            images,
            batch,
            output,
            confidenceThreshold);
        results.insert(results.end(), batchResults.begin(), batchResults.end());
    }

    // results[i] 对应第 i 张图片的有效检测结果集合。
    releaseModel(model);
    return 0;
}
