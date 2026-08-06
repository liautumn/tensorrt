# YOLO26 + TensorRT 11 极简多图推理

在 `main()` 中配置 `N` 张图片，程序同步输出每张图片的检测结果。程序从 engine profile 读取最大 batch，并自动拆批。例如最大 batch 是 4，输入 10 张图时只执行三轮：`4 + 4 + 2`。

模型要求：

- TensorRT 11
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

## 运行

在 [src/main.cpp](src/main.cpp) 开头修改 engine 和图片路径，然后运行：

```powershell
.\build\Release\yolo26_trt.exe
```

`main()` 只负责按顺序拼装学习步骤：

- `readEngine` / `initModel`：`src/model.cpp`
- `loadImages`：`src/image.cpp`
- `splitByMaxBatch`：`src/batch.cpp`
- `preprocessBatch`：`src/image.cpp`
- `setBatchSize` / `copyToGpu` / `infer` / `copyToCpu`：`src/inference.cpp`
- `printBatchResults`：`src/result.cpp`

在 CLion 中点击 `main()` 里的函数名即可跳到对应流程。代码没有异步队列或后台线程。
