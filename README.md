# YOLO26 TensorRT

只实现 Ultralytics YOLO26 端到端物体检测。核心接口同时支持单图和批量 `N`，不包含
旧模型兼容代码，也不包含分类、分割、姿态或旋转框分支。

## 模型协议

- 输入：`float32 [N,3,H,W]`，RGB、居中 letterbox、填充值 114、除以 255。
- 输出：`float32 [N,K,6]`，每行是
  `[x1,y1,x2,y2,confidence,class_id]`，默认 `K=300`。
- YOLO26 端到端输出已经完成候选框解码和 Top-K 选择。本项目只在 CPU 过滤置信度、
  恢复原图坐标，不再运行 CUDA NMS 或 CUDA 后处理核。
- 动态 engine 接受 profile 范围内的实际图片数 `N`；固定 batch engine 要求图片数完全一致。

推荐导出：

```bash
yolo export model=yolo26n.pt format=engine imgsz=640 batch=8 dynamic=True \
  quantize=16 end2end=True nms=False max_det=300 device=0
```

官方 Ultralytics metadata 前缀和纯 TensorRT plan 都可以直接加载。

## 代码分层

- `include/`：公共接口和各模块头文件。
- `src/preprocess.cu`：CUDA letterbox、BGR 转 RGB、归一化和 HWC 转 CHW。
- `src/tensorrt_engine.cpp`：engine 加载、动态 shape、tensor 绑定和 `enqueueV3`。
- `src/postprocess.cpp`：CPU 置信度过滤和坐标恢复。
- `src/timer.cpp`：预处理、推理、后处理和总耗时。
- `src/logger.cpp`：控制台日志和每日 `yyyy-MM-dd-trt.log`。
- `include/cpm.h`：多线程请求自动组成 batch 的 CPM 队列。
- `workspace/export/`：生成 DLL/SO 的同步 C ABI 和 CPM C ABI。
- `workspace/example/`：由 `config.h` 驱动的 C++ 示例，无命令行参数。
- `doc/C#/`：由 `Config.cs` 驱动的 C# P/Invoke 示例。

## 构建动态库

Linux：

```bash
cmake -S . -B build/linux -C doc/cmake/CMakeLists.linux \
  -DTENSORRT_ROOT=/opt/TensorRT-10.13.3.9
cmake --build build/linux --parallel
```

生成 `workspace/export/libyolo26.so` 和可选的 `yolo26_example`。Windows x64：

```powershell
cmake -S . -B build/win -C doc/cmake/CmakeLists.win `
  -DTENSORRT_ROOT=C:/TensorRT-10.13.3.9 -DOpenCV_DIR=C:/opencv/build
cmake --build build/win --config Release --parallel
```

生成 `workspace/export/yolo26.dll` 和 `workspace/export/yolo26.lib`。只需要动态库时添加
`-DYOLO26_BUILD_EXAMPLE=OFF`，核心和 C ABI 均不依赖 OpenCV。

## 配置与调用

C++ 示例参数集中在 `workspace/example/config.h`。将 engine 路径和图片列表改好后直接运行：

```bash
./build/linux/yolo26_example
```

C# 示例参数集中在 `doc/C#/TestDLL/TestDLL/TensorRT/Config.cs`。同步接口
`Yolo26Detector.Predict` 显式接收 `N` 张图片；`Yolo26Cpm` 可让多个线程逐张提交，native
队列自动组成 batch。CPM 要求 engine 的最小 batch 为 1，配置上限不能超过 engine profile。
原始图像通过 BGR 指针、宽、高、stride 传递，不跨 DLL 传递 `cv::Mat*`，结果内存由
SafeHandle 管理。

每次推理会在控制台和当日 `yyyy-MM-dd-trt.log` 中记录：图片数 `N`、预处理、TensorRT
推理、后处理和总耗时。

## Docker

```bash
doc/docker/build.sh
doc/docker/launch.sh
cmake -S . -B build/docker -C doc/cmake/CMakeLists.docker
cmake --build build/docker --parallel
```

默认镜像使用 CUDA 12.9.1 和 TensorRT 10.13.3.9。需要 NVIDIA Container Toolkit 才能在
容器中访问 GPU。

参考：[YOLO26 模型文档](https://docs.ultralytics.com/models/yolo26/) 和
[Ultralytics 导出输出说明](https://docs.ultralytics.com/modes/export/#what-do-the-output-tensors-represent-in-exported-yolo-models)。
