// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

// 引入批次模块：提供 Batch、Batches 和 splitByMaxBatch()。
#include "batch.h"
// 引入图片模块：提供 Images 和 loadImages()。
#include "image.h"
// 引入推理模块：提供 setBatchSize()、infer() 和 copyToCpu()。
#include "inference.h"
// 引入模型模块：提供 EngineData、Model、readEngine()、initModel() 和 releaseModel()。
#include "model.h"
// 引入 CUDA letterbox 预处理模块和仿射矩阵类型。
#include "preprocess.h"
// 引入结果模块：提供 Detection、Results、printBatchResults() 和 showResults()。
#include "result.h"
// 引入 CUDA Event 计时器：用于测量 TensorRT 推理的 GPU stream 耗时。
#include "timer.h"

// std::chrono::steady_clock 用于测量完整 CUDA 预处理墙钟耗时和 CPU 后处理耗时。
#include <chrono>
// std::fixed 和 std::setprecision() 用于将耗时固定显示为 3 位小数。
#include <iomanip>
// std::cout 只在控制台打印每个批次的三段耗时，不写入日志文件。
#include <iostream>
// std::string 用于保存 engine 路径和图片路径。
#include <string>
// std::vector 用于保存多张图片路径以及 CPU 输出数组。
#include <vector>

// C++ 程序入口；本函数只负责按照执行顺序拼装各个学习模块。
int main()
{
    // enginePath：传给 readEngine() 的 TensorRT engine 文件路径。
    std::string const enginePath = "D:/autumn/Documents/CLionProjects/tensorrt/model/win.engine";
    // imagePaths：传给 loadImages() 的图片路径集合；元素数量就是待推理图片数量。
    std::vector<std::string> const imagePaths{
        // 第 0 张待推理图片的路径。
        "D:/autumn/Documents/CLionProjects/tensorrt/model/1.jpg"
    };
    // confidenceThreshold：传给 printBatchResults() 的最低置信度；低于 0.7 的框会被过滤。
    float const confidenceThreshold = 0.7F;

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
    // 复用同一对 CUDA Event，避免每轮推理重复创建和销毁事件。
    trt_timer::Timer inferenceTimer;

    // 依次处理拆分后的每个批次；例如 10 张图、maxBatch=4 时依次得到 4、4、2。
    for (Batch const& batch : batches)
    {
        // 把当前实际图片数量 batch.size 写入 TensorRT context 的动态输入 shape。
        setBatchSize(model, batch.size);

        while (true)
        {
            // 记录当前批次预处理开始时间；steady_clock 不受系统时间调整影响。
            auto const preprocessStart = std::chrono::steady_clock::now();
            // 上传当前批次原图并执行 CUDA letterbox，直接生成 [N,3,H,W] FP32 模型输入。
            AffineMatrices affineMatrices = preprocessBatchToGpu(
                // model：提供 CUDA stream、输入显存、目标尺寸和可复用 workspace。
                model,
                // images：全部已经读取的原始图片。
                images,
                // batch：当前批次的起始图片下标和实际图片数量。
                batch);
            // 记录当前批次预处理结束时间。
            auto const preprocessEnd = std::chrono::steady_clock::now();

            // 在 model.stream 上记录起始事件；它位于 CUDA 预处理同步完成之后、enqueueV3 之前。
            inferenceTimer.start(model.stream);
            // 使用 model 中的 context 和 CUDA stream 调用 enqueueV3，并同步等待推理完成。
            infer(model);
            // 从 GPU 输出显存复制本批次结果到 CPU；返回数组 shape 为 [batch,max_det,6]。
            std::vector<float> output = copyToCpu(model, batch.size);
            // 沿用当前计时顺序：同步 D2H 完成后才提交结束事件，因此该值并非严格的纯 GPU compute。
            // false 表示 Timer 不单独输出，后面会与预处理和后处理耗时统一打印。
            float const inferenceMilliseconds = inferenceTimer.stop("inference", false);

            // 记录后处理开始时间；后处理包含过滤、坐标还原和结果集合构建。
            auto const postprocessStart = std::chrono::steady_clock::now();
            // 解析并过滤当前批次结果，同时返回按图片分组的有效检测集合。
            Results batchResults = printBatchResults(
                // model：提供每张图的最大候选框数量。
                model,
                // batch：提供本批次实际图片数以及结果对应的全局起点。
                batch,
                // affineMatrices：CUDA letterbox 生成的网络坐标到原图坐标逆变换。
                affineMatrices,
                // output：copyToCpu() 返回的 [batch,max_det,6] 原始浮点输出。
                output,
                // confidenceThreshold：只保留置信度大于等于该值的检测框。
                confidenceThreshold);
            // 将当前 batch 的分组结果追加到总结果中，并保持与输入图片相同的顺序。
            results.insert(results.end(), batchResults.begin(), batchResults.end());
            // 记录当前批次后处理结束时间。
            auto const postprocessEnd = std::chrono::steady_clock::now();

            // 计算预处理耗时；duration<double, milli> 把时间差转换为毫秒浮点数。
            double const preprocessMilliseconds
                = std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();
            // 计算后处理耗时；其中包含检测结果的控制台输出和日志写入。
            double const postprocessMilliseconds
                = std::chrono::duration<double, std::milli>(postprocessEnd - postprocessStart).count();

            // timing 仅打印到控制台，不调用 Validator::info，因此不会进入每日日志。
            auto const previousFlags = std::cout.flags();
            auto const previousPrecision = std::cout.precision();
            std::cout << std::fixed << std::setprecision(3)
                << "timing: batch=" << batch.size
                << " preprocess=" << preprocessMilliseconds << " ms"
                << " inference=" << inferenceMilliseconds << " ms"
                << " postprocess=" << postprocessMilliseconds << " ms"
                << " total=" << preprocessMilliseconds + inferenceMilliseconds + postprocessMilliseconds << " ms\n";
            std::cout.flags(previousFlags);
            std::cout.precision(previousPrecision);
        }
    }

    // 释放 initModel() 创建的显存、CUDA stream、context、engine 和 runtime。
    // 弹窗等待只依赖 CPU 端的 images 和 results，所以先释放 GPU 资源，避免查看结果时持续占用显存。
    releaseModel(model);
    // results[i] 对应 images[i]；从汇总结果中读取框，在原图副本上绘制后通过 OpenCV 弹窗显示。
    // 此调用位于全部批次的计时打印之后，绘制、窗口刷新和按键等待不会进入现有耗时统计。
    showResults(images, results);
    // 返回 0 表示程序正常结束。
    return 0;
}
