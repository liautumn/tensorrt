cmake_minimum_required(VERSION 3.28)
project(yolo CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 查找 OpenCV 库
set(OpenCV_DIR "D:/cuda/opencv/build")
find_package(OpenCV REQUIRED)

# 设置包含路径
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        "D:/cuda/TensorRT-8.6.1.6/include"
        "D:/cuda/TensorRT-8.6.1.6/samples/common"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
)

# 设置库路径
link_directories(
        "D:/cuda/TensorRT-8.6.1.6/lib"
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
        src/yolo.cu
        src/yolo.hpp
        src/main.cpp
        src/Config.cpp
        src/exportOnceAsync.cpp
        src/exportOnceSync.cpp
)

#add_executable(${PROJECT_NAME}
#        src/cpm.hpp
#        src/infer.cu
#        src/infer.hpp
#        src/yolo.cu
#        src/yolo.hpp
#        src/main.cpp
#        src/Config.cpp
#        src/exportOnceAsync.cpp
#        src/exportOnceSync.cpp
#)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# 链接库
target_link_libraries(${PROJECT_NAME} ${LIBS})