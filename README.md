# YOLO26 + TensorRT 11 极简多图推理

在 `main()` 中配置 `N` 张图片，程序同步生成每张图片的检测结果。程序从 engine profile 读取最大 batch，并自动拆批。例如最大 batch 是 4，输入 10 张图时会拆成：`4 + 4 + 2`。

模型要求：

- TensorRT 11
- 支持编译 CUDA `.cu` 文件的 CUDA Toolkit
- 动态 batch 输入 `[-1, 3, H, W]`
- profile 的最小 batch 必须是 1
- YOLO end-to-end 输出 `[N, max_det, 6]`

## 生成 Engine

下面示例把最大 batch 设置为 4：

```powershell
trtexec.exe `
  --onnx=best.onnx `
  --saveEngine=best.engine `
  --minShapes=images:1x3x640x640 `
  --optShapes=images:4x3x640x640 `
  --maxShapes=images:4x3x640x640
```

代码直接读取 `trtexec` 生成的纯 TensorRT plan。

## 配置

在 [CMakeLists.txt](CMakeLists.txt) 中设置 CUDA、OpenCV 和 TensorRT 路径，或在配置时传入：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" `
  -DOpenCV_DIR="C:/opencv/build" `
  -DTENSORRT_ROOT="C:/TensorRT-11.2.1"
cmake --build build --config Release
```

使用 Visual Studio generator 时必须安装对应 CUDA 版本的 Visual Studio Integration。
机器上存在多个 CUDA Toolkit 时，可在首次配置时增加 `-T "cuda=13.3"`；使用 CLion/Ninja 时则通过
`CUDAToolkit_ROOT` 或 `CMAKE_CUDA_COMPILER` 选择对应的 `nvcc.exe`。

## 运行

在 [src/main.cpp](src/main.cpp) 开头修改 engine 和图片路径，然后运行：

```powershell
.\build\Release\yolo26_trt.exe
```

预处理按当前推理 batch 执行：原始 BGR 图片先复制到 pinned memory，再通过
`cudaMemcpyAsync` 上传，CUDA kernel 完成保持宽高比的 letterbox、114 填充、双线性插值、
BGR 到 RGB 和 `1/255` 归一化，并直接写入 FP32 NCHW 输入显存。后处理使用同一组逆仿射矩阵还原检测框。

`main()` 只负责按顺序拼装学习步骤：

- `readEngine` / `initModel`：`src/model.cpp`
- `loadImages`：`src/image.cpp`
- `splitByMaxBatch`：`src/batch.cpp`
- `preprocessBatchToGpu`：`src/preprocess.cu`
- `setBatchSize` / `infer` / `copyToCpu`：`src/inference.cpp`
- `printBatchResults`：`src/result.cpp`

在 CLion 中点击 `main()` 里的函数名即可跳到对应流程。H2D 虽通过 CUDA stream 异步提交，
但预处理函数返回前会同步该 stream；当前代码没有跨批次异步流水线或后台线程。

当前 `main()` 中保留了用于连续测速的 `while (true)`，因此运行时会重复处理第一个 batch；
需要实际执行完整的 `4 + 4 + 2` 拆批流程时，应先移除该循环。
