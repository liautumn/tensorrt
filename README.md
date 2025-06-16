# YOLOV8+

**support**

* detect
* cls
* obb
* pose
* seg

**environment**

* TensorRT-10
* cuda 12.8
* cuDNN 9.8

```text
cmake -S . -B build
cmake --build build --config Release
cmake --build build --config Debug
cmake --build build --config Release --parallel 4  # 使用 4 个线程
```
