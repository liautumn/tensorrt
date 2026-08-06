// 防止当前头文件在同一个编译单元中被重复包含。
#pragma once

// 引入 Model 结构体，下面的函数通过它访问 TensorRT context、CUDA stream 和显存。
#include "model.h"

// 引入 std::vector；copyToCpu 使用它返回 CPU 端的模型输出。
#include <vector>

// 设置本轮推理实际使用的 batch 数量。
// model：已经初始化的模型对象，函数会修改其中 inputShape，并把新形状设置给 context。
// batchSize：本轮图片数量，必须在模型配置允许的 batch 范围内。
// 返回值：无；设置结果直接保存在 model.inputShape 和 model.context 中。
void setBatchSize(Model& model, int batchSize);

// 使用当前 context、输入形状和输入显存执行一次同步推理。
// model：已经设置好 batch、输入地址、输出地址和 CUDA stream 的模型对象。
// 返回值：无；推理结果写入 model.outputDevice，执行失败时抛出异常。
void infer(Model& model);

// 把本轮模型输出从 GPU 显存复制回 CPU。
// model：已经完成推理的模型对象，model.outputDevice 是输出显存地址。
// batchSize：本轮图片数量，用于计算本轮实际需要复制多少个输出元素。
// 返回值：一维 float 集合，按“图片、检测框、6 个字段”的顺序连续存放结果。
std::vector<float> copyToCpu(Model const& model, int batchSize);
