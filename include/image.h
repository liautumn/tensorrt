// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

// 防止同一个头文件在一次编译过程中被重复包含。
#pragma once

// 引入 OpenCV 的 cv::Mat；这里只用它保存从磁盘读取的原始 BGR 图片。
#include <opencv2/core/mat.hpp>

// 提供 std::string，用来保存图片文件路径。
#include <string>
// 提供 std::vector，用来保存多条路径或多张图片。
#include <vector>

// Images 是 std::vector<cv::Mat> 的简短别名，表示按输入顺序保存的多张图片。
using Images = std::vector<cv::Mat>;

// 按顺序读取多张图片。
// 参数 imagePaths：图片路径集合，以 const 引用传入，函数只读取而不复制、不修改它。
// 返回值：读取成功的图片集合；返回图片的顺序与 imagePaths 完全一致。
// 异常：任意图片读取失败时抛出 std::runtime_error，不返回不完整的结果。
Images loadImages(std::vector<std::string> const& imagePaths);
