cmake_minimum_required(VERSION 3.28)
project(yolo CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 定义头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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

#add_library(${PROJECT_NAME} SHARED
#        include/cpm.h
#        include/yolo.h
#        include/infer.h
#        src/infer.cu
#        src/yolo.cu
#        src/export/exportOnceSync.cpp
#        src/export/exportOnceAsync_old.cpp
#        src/export/exportBatch.cpp
#        src/export/exportOnceAsync_new.cpp
#)

add_executable(${PROJECT_NAME}
        include/cpm.h
        include/yolo.h
        include/infer.h
        src/infer.cu
        src/yolo.cu
        src/demo/config.h
        src/demo/main.cpp
        src/demo/barchDemo.cpp
        src/export/exportOnceSync.cpp
        src/export/exportOnceAsync_old.cpp
        src/export/exportBatch.cpp
        src/export/exportOnceAsync_new.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# 链接库
target_link_libraries(${PROJECT_NAME} ${LIBS})