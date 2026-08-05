# 要求 CMake 至少为 3.21，该版本开始支持 Visual Studio 2022 generator。
cmake_minimum_required(VERSION 3.21)
# 定义项目名称为 yolo26_trt，并声明这里只编译 C++。
project(yolo26_trt LANGUAGES CXX)

# 使用 C++17，便于使用 RAII、智能指针等标准库功能。
# 指定 C++ 标准版本为 17。
set(CMAKE_CXX_STANDARD 17)
# 如果编译器不支持 C++17，直接配置失败，不允许自动降低版本。
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 关闭 GNU 私有扩展，让代码使用标准 C++17。
set(CMAKE_CXX_EXTENSIONS OFF)

# ==================== Windows 路径配置：通常只需要修改下面三行 ====================
# Windows 路径在 CMake 中建议统一使用正斜杠 /，这样不需要处理反斜杠转义。
if(WIN32)
    # CUDA 13.3 默认安装目录；必须指向包含 include 和 lib/x64 的 CUDA 根目录。
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"
        CACHE PATH "CUDA Toolkit installation directory")
    # 必须指向包含 OpenCVConfig.cmake 的目录；官方 Windows 包通常是 C:/opencv/build。
    set(OpenCV_DIR "C:/opencv/build"
        CACHE PATH "Directory containing OpenCVConfig.cmake")
    # TensorRT zip 解压目录；其中应当包含 include/NvInfer.h 和 lib/nvinfer_11.lib。
    set(TENSORRT_ROOT "C:/TensorRT-11.2.1"
        CACHE PATH "TensorRT installation directory")
else()
    # 非 Windows 环境仍允许通过 -D 参数指定路径，不强制写死系统目录。
    set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation directory")
endif()

# 配置时打印用户指定的三个根目录，路径写错时可以直接从 CMake 输出中发现。
message(STATUS "CUDAToolkit_ROOT = ${CUDAToolkit_ROOT}")
message(STATUS "OpenCV_DIR        = ${OpenCV_DIR}")
message(STATUS "TENSORRT_ROOT     = ${TENSORRT_ROOT}")

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
# 查找 TensorRT 核心导入库；Windows 11.x 通常叫 nvinfer_11.lib，Linux 通常叫 libnvinfer.so。
find_library(TENSORRT_LIBRARY
    # NAMES 列出 Windows 和 Linux 可能使用的库名，不需要写 .lib 或 .so 后缀。
    NAMES nvinfer_11 nvinfer
    # 同样优先使用用户指定的 TensorRT 根目录。
    HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
    # TensorRT Windows zip 通常使用 lib；其他安装方式也可能使用 lib/x64 或 lib64。
    PATH_SUFFIXES lib lib/x64 lib64)

# 任意一个查找结果为空都无法编译。
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    # 给出明确错误，并告诉用户如何指定 TensorRT 路径。
    message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=/path/to/TensorRT-11.2.1")
endif()

# 打印最终找到的 TensorRT 头文件目录和库文件，便于核对是否误用了其他版本。
message(STATUS "TensorRT include   = ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library   = ${TENSORRT_LIBRARY}")
# OpenCV_VERSION 和 CUDAToolkit_VERSION 由 find_package 自动提供。
message(STATUS "OpenCV version     = ${OpenCV_VERSION}")
message(STATUS "CUDA version       = ${CUDAToolkit_VERSION}")

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
