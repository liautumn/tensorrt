# YOLO26 + TensorRT 11

模型要求：

- TensorRT11 CUDA13 C++20 
- 支持编译 CUDA `.cu` 文件的 CUDA Toolkit
- 包含 `core`、`imgcodecs`、`imgproc` 和 `highgui` 模块的 OpenCV
- 动态 batch 输入 `[-1, 3, H, W]`
- profile 的最小 batch 必须是 1
- YOLO end-to-end 输出 `[N, max_det, 6]`

## C++20 与资源所有权

项目同时以 C++20 和 CUDA C++20 编译。`Model` 是不可复制、不可移动的 RAII 类型，通过
`Model&` 显式传给初始化、预处理、推理和后处理函数。`initModel(model, engineData)` 会先
释放该实例已有的模型资源，再加载新 Engine，因此也可以分别初始化和使用多个 `Model` 实例。
TensorRT runtime、engine、execution context，以及 CUDA stream、event、device memory 和
pinned memory 都由 `std::unique_ptr` 与对应 deleter 管理。正常退出或初始化、推理过程中抛出
异常时，资源都会按依赖顺序自动释放，不需要手动调用清理函数。

只读连续数据接口使用 `std::span` 表达非拥有关系，路径使用 `std::filesystem::path`；实现中
同时使用 ranges、指定初始化器、`[[nodiscard]]` 和 `constexpr` 等现代 C++ 写法。

## 所有权与许可

Copyright (C) 2026 liqiuzhuang (GitHub: liautumn) and contributors

本项目中由 liqiuzhuang 创作的原创源代码和文档，其著作权归 liqiuzhuang
所有，并依据 GNU General Public License version 3 授权使用；该授权不构成
所有权转让。第三方材料仍归各自权利人所有。具体范围见 [NOTICE](NOTICE)，
完整许可条款见 [LICENSE](LICENSE)。

## 生成 Engine

```powershell
# Windows
trtexec.exe `
  --onnx=model/yolo26n.onnx `
  --saveEngine=model/win.engine `
  --minShapes=images:1x3x640x640 `
  --optShapes=images:1x3x640x640 `
  --maxShapes=images:1x3x640x640
```

```bash
# Linux
trtexec \
  --onnx=model/yolo26n.onnx \
  --saveEngine=model/linux.engine \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640
```

代码直接读取 `trtexec` 生成的纯 TensorRT plan。

## 运行

在 [src/main.cpp](src/main.cpp) 开头修改 engine 和图片路径，然后运行：

预处理按当前推理 batch 执行：原始 BGR 图片先复制到 pinned memory，再通过
`cudaMemcpyAsync` 上传，CUDA kernel 完成保持宽高比的 letterbox、114 填充、双线性插值、
BGR 到 RGB 和 `1/255` 归一化，并直接写入 FP32 NCHW 输入显存。后处理使用同一组逆仿射矩阵还原检测框。

所有图片处理完成后会为每张图片创建一个结果窗口。框和 `class`、`score` 标签直接来自
`results[i]`，绘制发生在原图副本上，不会修改原始图片或检测数据。在任意结果窗口按键后，
程序会关闭全部窗口并退出。绘制、窗口刷新和按键等待均位于耗时统计之外。

`main()` 只负责按顺序拼装：

- `readEngine` / `initModel`：`src/model.cpp`
- CUDA memory、stream 和 event 的 RAII 管理：`src/cuda_raii.cpp`
- `loadImages`：`src/image.cpp`
- `splitByMaxBatch`：`src/batch.cpp`
- `preprocessBatchToGpu`：`src/preprocess.cu`
- `setBatchSize` / `infer` / `copyToCpu`：`src/inference.cpp`
- `printBatchResults` / `showResults`：`src/result.cpp`

在 CLion 中点击 `main()` 里的函数名即可跳到对应流程。H2D 虽通过 CUDA stream 异步提交，
但预处理函数返回前会同步该 stream；当前代码没有跨批次异步流水线或后台线程。
`main()` 返回时 `Model` 会自动释放全部 TensorRT 和 CUDA 资源。

当前 `main()` 会按顺序执行全部批次，并将每个批次的结果追加到全局 `results`；因此
`results[i]` 始终与 `images[i]` 一一对应，显示阶段不需要再次读取或解析 TensorRT 输出。

# Linux 环境配置

## 1. 安装 C/C++ 开发工具链

### 更新软件源

```bash
sudo apt update
```

### 安装 C/C++ 编译环境

`build-essential` 包含：

- gcc
- g++
- make
- libc 开发文件
- 常用编译工具

```bash
sudo apt install build-essential
```

验证：

```bash
gcc --version
g++ --version
make --version
```

---

### 安装调试工具

#### Valgrind

用于：

- 内存泄漏检测
- 内存错误分析
- 性能分析

```bash
sudo apt install valgrind
```

#### GDB

GNU 调试器，用于：

- C/C++ 程序断点调试
- 崩溃分析
- 查看堆栈信息

```bash
sudo apt install gdb
```

验证：

```bash
gdb --version
```

---

### 安装 CMake

用于 C/C++ 项目构建：

```bash
sudo apt install cmake
```

验证：

```bash
cmake --version
```

---

### 安装 Git

用于代码管理：

```bash
sudo apt install git
```

验证：

```bash
git --version
```

---

## 2. 安装 OpenCV 开发库

安装 OpenCV C++ 开发环境：

```bash
sudo apt install libopencv-dev
```

验证：

```bash
pkg-config --modversion opencv4
```

---

# 3. 安装 Nvidia 驱动 和 CUDA

官方安装地址：

- Nvidia 驱动  
  https://www.nvidia.cn/geforce/drivers

- CUDA Toolkit  
  https://developer.nvidia.com/cuda-downloads


安装完成后确认：

```bash
nvidia-smi
nvcc --version
```

---

# 4. 安装 TensorRT

官方安装地址：

- TensorRT 11.x  
  https://developer.nvidia.com/tensorrt/download/11x


解压示例：

```bash
tar -xf TensorRT-11.2.1.2.tar.gz
```

假设安装目录：

```text
/home/autumn/dev/TensorRT-11.2.1.2
```

---

# 5. 配置环境变量

编辑用户环境：

```bash
nano ~/.bashrc
```

添加：

```bash
# ==========================
# CUDA
# ==========================
export CUDA_HOME=/usr/local/cuda-13.3

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDACXX=$CUDA_HOME/bin/nvcc


# ==========================
# TensorRT
# ==========================
export TENSORRT_PATH=/home/autumn/dev/TensorRT-11.2.1.2

export PATH=$TENSORRT_PATH/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_PATH/lib:$LD_LIBRARY_PATH
```

保存退出：

```
Ctrl + O
Enter
Ctrl + X
```

---

# 6. 重新加载环境变量

```bash
source ~/.bashrc
```

---

# 7. 验证安装

## CUDA

```bash
nvcc --version
```

示例：

```text
Cuda compilation tools, release 13.3
```

---

## TensorRT

查看版本：

```bash
trtexec --version
```

示例：

```text
TensorRT 11.2.1
```

---

## OpenCV

```bash
pkg-config --modversion opencv4
```

---

## 检查动态库

```bash
echo $LD_LIBRARY_PATH
```

应该包含：

```text
/usr/local/cuda-13.3/lib64
/home/autumn/dev/TensorRT-11.2.1.2/lib
```
