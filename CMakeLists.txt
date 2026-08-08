# SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
# SPDX-License-Identifier: GPL-3.0-only

# 要求 CMake 至少为 3.21
cmake_minimum_required(VERSION 3.21)

# ============================================================
# 第三方库路径
# ============================================================

if (WIN32)

    # CUDA Toolkit 根目录
    set(CUDAToolkit_ROOT
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"
            CACHE PATH
            "CUDA Toolkit installation directory"
    )

    # OpenCVConfig.cmake 所在目录
    set(OpenCV_DIR
            "D:/dev/opencv/build/x64/vc16/lib"
            CACHE PATH
            "Directory containing OpenCVConfig.cmake"
    )

    # TensorRT 根目录
    set(TENSORRT_ROOT
            "D:/dev/TensorRT-11.2.1.2"
            CACHE PATH
            "TensorRT installation directory"
    )

else ()

    # Linux CUDA Toolkit
    set(CUDAToolkit_ROOT
            "/usr/local/cuda-13.3"
            CACHE PATH
            "CUDA Toolkit installation directory"
    )

    # nvcc
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")

    # TensorRT
    set(TENSORRT_ROOT
            "/home/autumn/dev/TensorRT-11.2.1.2"
            CACHE PATH
            "TensorRT installation directory"
    )

endif ()

# 定义项目名称，同时启用 C++ 和 CUDA 编译语言
project(yolo26_trt
        LANGUAGES CXX CUDA
)

# ============================================================
# C++ / CUDA 标准配置
# ============================================================

# 使用 C++20
set(CMAKE_CXX_STANDARD 20)
# 编译器不支持 C++20 时直接失败
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 关闭 GNU 私有扩展
set(CMAKE_CXX_EXTENSIONS OFF)


# CUDA 使用 C++20
set(CMAKE_CUDA_STANDARD 20)
# CUDA 不支持 C++20 时直接失败
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 关闭 CUDA 非标准扩展
set(CMAKE_CUDA_EXTENSIONS OFF)


# CUDA 架构
# 75 : RTX20
# 86 : RTX30
# 89 : RTX40
# 120 : RTX50
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 120)

# ============================================================
# 查找 CUDA / OpenCV
# ============================================================

# CUDA Runtime
find_package(CUDAToolkit REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS
        core
        imgcodecs
        imgproc
        highgui
)

# ============================================================
# 查找 TensorRT
# ============================================================

# 查找 NvInfer.h
find_path(
        TENSORRT_INCLUDE_DIR
        NvInfer.h

        HINTS
        ${TENSORRT_ROOT}
        $ENV{TENSORRT_ROOT}

        PATH_SUFFIXES
        include
)

# 查找 TensorRT 库
find_library(
        TENSORRT_LIBRARY

        NAMES
        nvinfer_11
        nvinfer

        HINTS
        ${TENSORRT_ROOT}
        $ENV{TENSORRT_ROOT}

        PATH_SUFFIXES
        lib
        lib/x64
        lib64
)

# TensorRT 检查
if (NOT TENSORRT_INCLUDE_DIR)

    message(FATAL_ERROR
            "TensorRT include directory not found"
    )

endif ()

if (NOT TENSORRT_LIBRARY)

    message(FATAL_ERROR
            "TensorRT library not found"
    )

endif ()

# ============================================================
# 创建目标
# ============================================================

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

# ============================================================
# MSVC UTF-8
# ============================================================

if (MSVC)

    target_compile_options(
            yolo26_trt

            PRIVATE

            # C++ 文件 UTF-8
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>


            # CUDA 文件传递给 MSVC
            $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=/utf-8
            >

    )

endif ()

# ============================================================
# Include
# ============================================================

# 当前项目头文件
target_include_directories(
        yolo26_trt

        PRIVATE

        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# TensorRT 和 OpenCV 第三方头文件
# TensorRT 可能产生 deprecated warning，因此作为 SYSTEM
target_include_directories(
        yolo26_trt

        SYSTEM

        PRIVATE

        ${TENSORRT_INCLUDE_DIR}

        ${OpenCV_INCLUDE_DIRS}
)

# ============================================================
# Link
# ============================================================

target_link_libraries(
        yolo26_trt

        PRIVATE

        # TensorRT 核心推理库
        ${TENSORRT_LIBRARY}


        # CUDA Runtime
        CUDA::cudart


        # OpenCV
        ${OpenCV_LIBS}
)

# ============================================================
# CUDA 属性
# ============================================================

set_target_properties(
        yolo26_trt

        PROPERTIES

        CUDA_SEPARABLE_COMPILATION ON
)