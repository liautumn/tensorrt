cmake_minimum_required(VERSION 3.28)
project(tensorrt CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(OpenCV_DIR $ENV{OPENCV_PATH})
find_package(OpenCV REQUIRED)
set(TensorRT_DIR $ENV{TENSORRT_PATH})
set(CUDA_DIR $ENV{CUDA_PATH})

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_DIR}/include
        ${TensorRT_DIR}/samples/common
        ${CUDA_DIR}/include
)

link_directories(
        ${TensorRT_DIR}/lib
        ${CUDA_DIR}/lib/x64
)

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
#        src/export/exportMultiple.cpp
#        src/export/exportMultipleCpm.cpp
#        src/export/exportSingle.cpp
#        src/export/exportSingleCpm.cpp
#)

add_executable(${PROJECT_NAME}
        include/cpm.h
        include/yolo.h
        src/yolo.cpp
        include/infer.h
        src/infer.cu
        include/preprocess.cuh
        src/preprocess.cu
        include/postprocess.cuh
        src/postprocess.cu
        src/example/config.h
        src/example/main.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} ${LIBS})