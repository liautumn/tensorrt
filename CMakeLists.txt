# 要求本机 CMake 版本至少为 3.18，因为 CUDAToolkit package 从该版本开始稳定提供。
cmake_minimum_required(VERSION 3.18)
# 定义项目名称为 yolo26_trt，并声明这里只编译 C++。
project(yolo26_trt LANGUAGES CXX)

# 使用 C++17，便于使用 RAII、智能指针等标准库功能。
# 指定 C++ 标准版本为 17。
set(CMAKE_CXX_STANDARD 17)
# 如果编译器不支持 C++17，直接配置失败，不允许自动降低版本。
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 关闭 GNU 私有扩展，让代码使用标准 C++17。
set(CMAKE_CXX_EXTENSIONS OFF)

# TensorRT tar 包不一定安装到系统目录，可通过该变量指定根目录。
# CACHE PATH 让用户可以通过 -DTENSORRT_ROOT=/xxx 从命令行赋值。
set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation directory")

# CUDA 负责显存和异步推理，OpenCV 负责图片读取与 letterbox 预处理。
# 查找 CUDA Toolkit；REQUIRED 表示找不到就停止配置。
find_package(CUDAToolkit REQUIRED)
# 只要求 OpenCV 的 core、图片读取和图片处理三个模块。
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

# TensorRT 官方 tar 包和系统包都不保证提供通用的 CMake package，直接查找头文件和库。
# 查找包含 NvInfer.h 的目录，结果保存到 TENSORRT_INCLUDE_DIR。
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    # 优先查找 CMake 参数 TENSORRT_ROOT 和同名环境变量。
    HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
    # 在上述根目录的 include 子目录中查找。
    PATH_SUFFIXES include)
# 查找 TensorRT 核心动态库 libnvinfer.so，结果保存到 TENSORRT_LIBRARY。
find_library(TENSORRT_LIBRARY nvinfer
    # 同样优先使用用户指定的 TensorRT 根目录。
    HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
    # Linux 库通常位于 lib 或 lib64。
    PATH_SUFFIXES lib lib64)

# 任意一个查找结果为空都无法编译。
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    # 给出明确错误，并告诉用户如何指定 TensorRT 路径。
    message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=/path/to/TensorRT-11.2.1")
endif()

# 整个示例只有一个可执行文件，不额外拆分库。
# 把 src/main.cpp 编译成名为 yolo26_trt 的可执行程序。
add_executable(yolo26_trt src/main.cpp)
# 给 yolo26_trt 添加 TensorRT 和 OpenCV 头文件搜索目录。
target_include_directories(yolo26_trt PRIVATE
    # TensorRT 的 NvInfer.h 所在目录。
    ${TENSORRT_INCLUDE_DIR}
    # OpenCV 的 opencv2 头文件所在目录。
    ${OpenCV_INCLUDE_DIRS})
# 指定最终链接时需要的动态库。
target_link_libraries(yolo26_trt PRIVATE
    # TensorRT 核心推理库 libnvinfer。
    ${TENSORRT_LIBRARY}
    # CUDA Runtime 库，提供 cudaMalloc、cudaMemcpyAsync 等函数。
    CUDA::cudart
    # OpenCV core、imgcodecs 和 imgproc 库。
    ${OpenCV_LIBS})
