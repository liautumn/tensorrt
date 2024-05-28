# 设置 CMake 最低版本
cmake_minimum_required(VERSION 3.28)

# 定义项目名称
project(yolo CXX CUDA)

# 指定 C++ 编译标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 查找 OpenCV 库
set(OpenCV_DIR "D:/autumn/Desktop/cuda/opencv/build")
find_package(OpenCV REQUIRED)

# 设置包含路径
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        "D:/autumn/Desktop/cuda/TensorRT-8.6.1.6/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
)

# 设置库路径
link_directories(
        "D:/autumn/Desktop/cuda/TensorRT-8.6.1.6/lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64"
)

# 定义链接的库
set(LIBS
        ${OpenCV_LIBS}
        nvinfer
        nvinfer_plugin
        nvonnxparser
        cuda
        cublas
        cudart
        cudnn
)

add_library(${PROJECT_NAME} SHARED
        src/cpm.hpp
        src/infer.cu
        src/infer.hpp
        src/main.cpp
        src/yolo.cu
        src/yolo.hpp
        src/interfaceOnceSync.cpp
        src/interfaceOnceAsync.cpp
)

# 添加可执行文件
#add_executable(${PROJECT_NAME}
#        src/cpm.hpp
#        src/infer.cu
#        src/infer.hpp
#        src/main.cpp
#        src/yolo.cu
#        src/yolo.hpp
#        src/interfaceOnceSync.cpp
#        src/interfaceOnceAsync.cpp
#)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# 链接库
target_link_libraries(${PROJECT_NAME} ${LIBS})