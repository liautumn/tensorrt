// 引入批次模块：提供 Batch、Batches 和 splitByMaxBatch()。
#include "batch.h"
// 引入图片模块：提供 Images、loadImages() 和 preprocessBatch()。
#include "image.h"
// 引入推理模块：提供 setBatchSize()、copyToGpu()、infer() 和 copyToCpu()。
#include "inference.h"
// 引入模型模块：提供 EngineData、Model、readEngine()、initModel() 和 releaseModel()。
#include "model.h"
// 引入结果模块：提供 Detection、Results 和 printBatchResults()。
#include "result.h"

// std::string 用于保存 engine 路径和图片路径。
#include <string>
// std::vector 用于保存多张图片路径以及 CPU 输出数组。
#include <vector>

// C++ 程序入口；本函数只负责按照执行顺序拼装各个学习模块。
int main()
{
    // enginePath：传给 readEngine() 的 TensorRT engine 文件路径。
    std::string const enginePath = "C:\\Users\\autumn\\CLionProjects\\tensorrt\\model\\best.engine";
    // imagePaths：传给 loadImages() 的图片路径集合；元素数量就是待推理图片数量。
    std::vector<std::string> const imagePaths{
        // 第 0 张待推理图片的路径。
        "C:\\Users\\autumn\\CLionProjects\\tensorrt\\model\\1.jpg",
        // 第 1 张待推理图片的路径。
        "C:\\Users\\autumn\\CLionProjects\\tensorrt\\model\\2.jpg"
    };
    // confidenceThreshold：传给 printBatchResults() 的最低置信度；低于 0.25 的框会被过滤。
    float const confidenceThreshold = 0.25F;

    // 读取 enginePath 指向的二进制文件；返回值 engineData 是完整的 engine 字节数组。
    EngineData engineData = readEngine(enginePath);
    // 把 engineData 传给 TensorRT，创建 runtime、engine、context、CUDA stream 和显存。
    Model model = initModel(engineData);
    // 按 imagePaths 逐张读取图片；返回的 images[i] 与 imagePaths[i] 一一对应。
    Images images = loadImages(imagePaths);
    // 参数 1 是总图片数，参数 2 是模型单次最大 batch；返回按顺序拆好的批次集合。
    Batches batches = splitByMaxBatch(images.size(), model.maxBatch);
    // results 保存全部图片的有效检测结果；results[i] 最终对应 images[i]。
    Results results;
    // 只预留与图片数相同的外层容量，减少后续合并批次结果时的重复内存分配。
    results.reserve(images.size());

    // 依次处理拆分后的每个批次；例如 10 张图、maxBatch=4 时依次得到 4、4、2。
    for (Batch const& batch : batches)
    {
        // 预处理当前批次并生成模型输入 cv::Mat；内部数据排列为 [N,3,H,W] FP32。
        cv::Mat input = preprocessBatch(
            // images：全部已经读取的原始图片。
            images,
            // batch：当前批次的起始图片下标和实际图片数量。
            batch,
            // model.inputHeight：模型要求的输入高度。
            model.inputHeight,
            // model.inputWidth：模型要求的输入宽度。
            model.inputWidth);

        // 把当前实际图片数量 batch.size 写入 TensorRT context 的动态输入 shape。
        setBatchSize(model, batch.size);
        // 把预处理后的 CPU input 数据复制到 model.inputDevice 指向的 GPU 显存。
        copyToGpu(model, input);
        // 使用 model 中的 context 和 CUDA stream 调用 enqueueV3，并同步等待推理完成。
        infer(model);
        // 从 GPU 输出显存复制本批次结果到 CPU；返回数组 shape 为 [batch,max_det,6]。
        std::vector<float> output = copyToCpu(model, batch.size);
        // 解析、过滤并打印当前批次结果，同时返回按图片分组的有效检测集合。
        Results batchResults = printBatchResults(
            // model：提供输入尺寸和每张图最大候选框数量。
            model,
            // images：用于根据每张原图尺寸还原检测框坐标。
            images,
            // batch：用于确定本批次对应原图片集合中的哪些下标。
            batch,
            // output：copyToCpu() 返回的 [batch,max_det,6] 原始浮点输出。
            output,
            // confidenceThreshold：只保留置信度大于等于该值的检测框。
            confidenceThreshold);
        // 将当前 batch 的分组结果追加到总结果中，并保持与输入图片相同的顺序。
        results.insert(results.end(), batchResults.begin(), batchResults.end());
    }

    // results[i] 对应第 i 张图片的有效检测结果集合。
    // 释放 initModel() 创建的显存、CUDA stream、context、engine 和 runtime。
    releaseModel(model);
    // 返回 0 表示程序正常结束。
    return 0;
}
