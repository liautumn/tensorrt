# YOLO26 + TensorRT 11

模型要求：

- TensorRT 11
- 支持编译 CUDA `.cu` 文件的 CUDA Toolkit
- 包含 `core`、`imgcodecs`、`imgproc` 和 `highgui` 模块的 OpenCV
- 动态 batch 输入 `[-1, 3, H, W]`
- profile 的最小 batch 必须是 1
- YOLO end-to-end 输出 `[N, max_det, 6]`

## 生成 Engine

```powershell
trtexec.exe `
  --onnx=model/yolo26n.onnx `
  --saveEngine=model/win.engine `
  --minShapes=images:1x3x640x640 `
  --optShapes=images:1x3x640x640 `
  --maxShapes=images:1x3x640x640
  
  
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
- `loadImages`：`src/image.cpp`
- `splitByMaxBatch`：`src/batch.cpp`
- `preprocessBatchToGpu`：`src/preprocess.cu`
- `setBatchSize` / `infer` / `copyToCpu`：`src/inference.cpp`
- `printBatchResults` / `showResults`：`src/result.cpp`

在 CLion 中点击 `main()` 里的函数名即可跳到对应流程。H2D 虽通过 CUDA stream 异步提交，
但预处理函数返回前会同步该 stream；当前代码没有跨批次异步流水线或后台线程。

当前 `main()` 会按顺序执行全部批次，并将每个批次的结果追加到全局 `results`；因此
`results[i]` 始终与 `images[i]` 一一对应，显示阶段不需要再次读取或解析 TensorRT 输出。
