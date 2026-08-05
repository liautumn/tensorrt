# YOLO26 + TensorRT 11 极简 C++ 推理

这个示例只做三件事：OpenCV letterbox 预处理、TensorRT 11 批量推理、打印检测框。YOLO26 默认 one-to-one 检测头输出 `[N, max_det, 6]`，每行是 `[x1, y1, x2, y2, score, class_id]`，因此不需要 NMS。示例仅按置信度过滤，并把 letterbox 坐标还原到原图。

## 环境

- Windows 10/11 x64 + NVIDIA GPU
- CUDA 13.3 Update 1（TensorRT 11.2.1 官方包的构建版本）
- TensorRT 11.2.1
- OpenCV 4
- Visual Studio 2022（安装“使用 C++ 的桌面开发”）
- CMake 3.21+

TensorRT engine 与构建它的 TensorRT 版本、GPU 平台相关，建议直接在 Windows 部署机器上导出。

## 模型导出与量化

安装或更新导出工具。TensorRT 11 的 FP16/INT8 是 strongly typed 工作流，Ultralytics 会调用 NVIDIA ModelOpt，把精度或 Q/DQ 节点写进 ONNX 后再构建 engine：

```bash
python -m pip install -U ultralytics "nvidia-modelopt[onnx]>=0.44"
```

确认 Python 导出环境确实加载 TensorRT 11.2.1，而不是系统里的其他版本：

```bash
python -c "import tensorrt as trt; print(trt.__version__)"
```

FP32 基线：

```bash
yolo export model=yolo26n.pt format=engine imgsz=640 batch=8 dynamic=True end2end=True quantize=32 workspace=4 device=0
```

FP16：

```bash
yolo export model=yolo26n.pt format=engine imgsz=640 batch=8 dynamic=True end2end=True quantize=16 workspace=4 device=0
```

INT8 PTQ。`data` 应指向任务匹配的数据集 YAML，其中的验证集图片需要能代表真实部署分布，建议至少约 500 张：

```bash
yolo export model=yolo26n.pt format=engine imgsz=640 batch=8 dynamic=True end2end=True quantize=8 data=D:/datasets/my_dataset/data.yaml fraction=1.0 workspace=4 device=0
```

`batch=8` 是 engine 可接受的最大 batch。程序一次可输入任意 N 张图片：动态 batch engine 会自动按最多 8 张分批，静态 batch engine 的最后一批会自动补齐。

> TensorRT 11 已删除旧的 `trtexec --fp16`、`--int8`、`--calib` 等选项。不要把 TensorRT 10 的量化命令用于 11.2.1。INT8 engine 的精度和性能需要在目标 GPU 上用验证集重新评估。

## 配置 Windows 依赖路径

在 `CMakeLists.txt` 顶部修改这三个路径：

```cmake
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" ...)
set(OpenCV_DIR "C:/opencv/build" ...)
set(TENSORRT_ROOT "C:/TensorRT-11.2.1" ...)
```

也可以不修改文件，直接在 PowerShell 中覆盖：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" `
  -DOpenCV_DIR="C:/opencv/build" `
  -DTENSORRT_ROOT="C:/TensorRT-11.2.1"
```

## 编译

```powershell
cmake --build build --config Release
```

运行前把 TensorRT 和 OpenCV DLL 目录加入当前 PowerShell 的 `PATH`：

```powershell
$env:Path = "C:\TensorRT-11.2.1\lib;C:\opencv\build\x64\vc16\bin;$env:Path"
```

## 配置并运行 N 张图片

先在 `src/main.cpp` 的 `main()` 开头直接修改模型和图片地址：

```cpp
std::string const enginePath = "D:/models/yolo26n.engine";
std::vector<std::string> const imagePaths{
    "D:/images/1.jpg",
    "D:/images/2.jpg",
    "D:/images/3.jpg",
};
```

然后直接运行，不需要命令行参数：

```powershell
.\build\Release\yolo26_trt.exe
```

程序自动读取 engine 的输入尺寸和 batch profile，输出类别 ID、置信度和原图坐标。置信度阈值是 `src/main.cpp` 中的 `kConfidenceThreshold`，默认 `0.25`。

Ultralytics 生成的 `.engine` 在 TensorRT plan 前带有一段 JSON 元数据；本示例已自动识别并跳过该头，也兼容 `trtexec` 生成的纯 plan。

## 参考

- [TensorRT 11.2.1 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/11.2.1/getting-started/release-notes.html)
- [TensorRT 10.x 到 11.x 的 trtexec 迁移](https://docs.nvidia.com/deeplearning/tensorrt/11.2.1/api/migration/tensorrt-10x-to-11x-trtexec.html)
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics TensorRT 导出](https://docs.ultralytics.com/integrations/tensorrt/)
