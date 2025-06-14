cmake_minimum_required(VERSION 3.28)
project(tensorrt CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 定义头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(OpenCV_DIR $ENV{OPENCV_PATH})
find_package(OpenCV REQUIRED)
set(TensorRT_DIR $ENV{TENSORRT_PATH})
set(CUDA_DIR $ENV{CUDA_PATH})

# 设置包含路径
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_DIR}/include
        ${TensorRT_DIR}/samples/common
        ${CUDA_DIR}/include
)

# 设置库路径
link_directories(
        ${TensorRT_DIR}/lib
        ${CUDA_DIR}/lib/x64
)

add_library(${PROJECT_NAME} SHARED
        src/infer.cu
        src/yolo.cu
        src/export/export_single_cpm.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# 链接库
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        nvinfer
        cuda
        cublas
        cudart
        cudnn
)