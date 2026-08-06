// 引入本文件中四个推理步骤的函数声明，以及它们使用的 Model 类型。
#include "inference.h"

// 引入 std::runtime_error，用于在 TensorRT 提交推理失败时报告错误。
#include <stdexcept>

// 设置当前这一轮推理实际包含的图片数量。
// model 使用非常量引用，因为这里会修改输入形状和 execution context 的状态。
// batchSize 是当前批次的图片数，例如 10 张图按最大 batch 4 拆分时依次为 4、4、2。
void setBatchSize(Model& model, int batchSize)
{
    // inputShape 的第 0 维是 batch 维，把它改成本轮实际图片数量。
    model.inputShape.d[0] = batchSize;
    // 把新的完整输入形状交给 TensorRT context。
    // 第一个参数是输入张量名称的 C 字符串，第二个参数是包含新 batch 的输入维度。
    model.context->setInputShape(model.inputName.c_str(), model.inputShape);
}

// 在当前 CUDA stream 上执行一次 TensorRT 推理，并等待这次推理完成。
// model 中的 context 必须已经设置本轮 batch，输入显存中也必须已有预处理数据。
void infer(Model& model)
{
    // enqueueV3 把一次模型执行提交到指定 CUDA stream；返回 false 表示提交失败。
    if (!model.context->enqueueV3(model.stream))
    {
        // 抛出异常并停止当前流程，避免继续读取无效或未生成的输出数据。
        throw std::runtime_error("enqueueV3 failed");
    }
    // 等待这个 stream 中已提交的工作完成，使当前接口表现为同步推理。
    // 同步完成后，model.outputDevice 中才是本轮可读取的完整输出。
    cudaStreamSynchronize(model.stream);
}

// 把当前批次的推理结果从 GPU 输出显存复制到 CPU，并作为 vector 返回。
// model 只用于读取显存地址及输出尺寸信息，因此使用常量引用。
// batchSize 决定只取本轮真实图片的结果，而不是总按最大 batch 取结果。
std::vector<float> copyToCpu(Model const& model, int batchSize)
{
    // 为 CPU 输出申请连续空间。
    // 每张图片最多有 maxDetections 个候选框，每个框固定包含 6 个 float 字段：
    // x1、y1、x2、y2、confidence、classId。
    std::vector<float> output(
        // 转为 size_t 后计算元素数量：本批图片数 * 每图最大框数 * 每框字段数。
        static_cast<std::size_t>(batchSize) * model.maxDetections * 6);

    // 把 TensorRT 写入 GPU 的输出复制到刚刚申请的 CPU vector 中。
    cudaMemcpy(
        // 目标地址：CPU 端 vector 的首元素地址。
        output.data(),
        // 源地址：初始化模型时为输出张量申请的 GPU 显存。
        model.outputDevice,
        // 复制字节数：输出 float 元素数量乘以一个 float 的字节数。
        output.size() * sizeof(float),
        // 复制方向：从 GPU Device 复制到 CPU Host。
        cudaMemcpyDeviceToHost);
    // 按值返回 CPU 输出；调用方随后可以解析每张图片的检测框。
    return output;
}
