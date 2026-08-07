# SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
# SPDX-License-Identifier: GPL-3.0-only

# 要求 CMake 至少为 3.21，该版本开始支持 Visual Studio 2022 generator。
cmake_minimum_required(VERSION 3.21)
# 定义项目名称为 yolo26_trt；CUDA language 会在工具链路径配置完成后启用。
project(yolo26_trt LANGUAGES CXX)

# 使用 C++17，保证当前代码使用的标准库和语言特性可用。
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
    set(OpenCV_DIR "D:/dev/opencv/build/x64/vc16/lib"
        CACHE PATH "Directory containing OpenCVConfig.cmake")
    # TensorRT zip 解压目录；其中应当包含 include/NvInfer.h 和 lib/nvinfer_11.lib。
    set(TENSORRT_ROOT "D:/dev/TensorRT-11.2.1.2"
        CACHE PATH "TensorRT installation directory")
else()
    # 非 Windows 环境仍允许通过 -D 参数指定路径，不强制写死系统目录。
    set(CUDAToolkit_ROOT
            "/usr/local/cuda-13.3"
            CACHE PATH "CUDA Toolkit installation directory")

#    set(OpenCV_DIR
#            "/usr/lib/x86_64-linux-gnu/cmake/opencv4"
#            CACHE PATH "Directory containing OpenCVConfig.cmake")

    set(TENSORRT_ROOT
            "/home/autumn/dev/TensorRT-11.2.1.2"
            CACHE PATH "TensorRT installation directory")

    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.3/bin/nvcc")
endif()

# 启用 CUDA language，使 add_executable 中的 src/preprocess.cu 交给 nvcc 编译。
enable_language(CUDA)
# 指定 CUDA host/device 代码使用 C++17。
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
# 编译器不支持 CUDA C++17 时直接配置失败，不允许自动降低标准版本。
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 关闭 CUDA 编译器的非标准语言扩展，与普通 C++ 源文件设置保持一致。
set(CMAKE_CUDA_EXTENSIONS OFF)

# 配置时打印用户指定的三个根目录，路径写错时可以直接从 CMake 输出中发现。
message(STATUS "CUDAToolkit_ROOT = ${CUDAToolkit_ROOT}")
message(STATUS "OpenCV_DIR        = ${OpenCV_DIR}")
message(STATUS "TENSORRT_ROOT     = ${TENSORRT_ROOT}")

# CUDA 负责预处理 kernel、显存、数据复制和推理 stream，OpenCV 负责图片读取、画框和弹窗。
# 查找 CUDA Toolkit；REQUIRED 表示找不到就停止配置。
find_package(CUDAToolkit REQUIRED)
# 显式要求代码实际使用的模块：cv::Mat、图片读取、框/文字绘制以及 HighGUI 窗口。
# 若 OpenCV 缺少 highgui 等组件，配置阶段会直接给出错误，而不是等到链接时才失败。
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

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

add_executable(yolo26_trt
    src/main.cpp
    src/model.cpp
    src/validator.cpp
    src/batch.cpp
    src/image.cpp
    src/preprocess.cu
    src/inference.cpp
    src/timer.cpp
    src/result.cpp
)

# MSVC 默认按系统代码页读取源文件；项目和 CUDA 头文件使用 UTF-8。
# 对 CUDA 编译，-Xcompiler 将该选项传递给 nvcc 调用的 MSVC 主机编译器。
if(MSVC)
    target_compile_options(yolo26_trt PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# 给 yolo26_trt 添加 TensorRT 和 OpenCV 头文件搜索目录。
target_include_directories(yolo26_trt PRIVATE
    # 当前项目自己的 model.h、image.h 等头文件所在目录。
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # OpenCV 的 opencv2 头文件所在目录。
    ${OpenCV_INCLUDE_DIRS})
# TensorRT 头文件包含官方保留的旧 API 兼容层；作为系统头处理，避免其内部弃用告警污染项目构建输出。
target_include_directories(yolo26_trt SYSTEM PRIVATE
    ${TENSORRT_INCLUDE_DIR})
# 指定最终链接时需要的动态库。
target_link_libraries(yolo26_trt PRIVATE
    # TensorRT 核心推理库 libnvinfer。
    ${TENSORRT_LIBRARY}
    # CUDA Runtime 库，提供 cudaMalloc、cudaMemcpy、CUDA stream 等函数。
    CUDA::cudart
    # OpenCV 配置导出的库集合，为 cv::Mat、imread、rectangle、putText 和 imshow 提供实现。
    ${OpenCV_LIBS})
