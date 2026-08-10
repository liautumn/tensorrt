// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

// 防止同一个头文件在一次编译过程中被重复包含。
#pragma once

// 引入 OpenCV 的 cv::Mat；这里只用它保存从磁盘读取的原始 BGR 图片。
#include <opencv2/core/mat.hpp>

#include <filesystem>
#include <span>
// 提供 std::vector，用来保存多条路径或多张图片。
#include <vector>

// Images 是 std::vector<cv::Mat> 的简短别名，表示按输入顺序保存的多张图片。
using Images = std::vector<cv::Mat>;
using ImagePaths = std::vector<std::filesystem::path>;

// 按顺序读取多张图片。
// 参数 imagePaths：图片路径的只读非拥有视图，函数不会复制或修改路径集合。
// 返回值：读取成功的图片集合；返回图片的顺序与 imagePaths 完全一致。
// 异常：任意图片读取失败时抛出 std::runtime_error，不返回不完整的结果。
[[nodiscard]] Images loadImages(std::span<std::filesystem::path const> imagePaths);
