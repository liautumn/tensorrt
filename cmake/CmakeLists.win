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
        nvinfer_10
        nvinfer_plugin_10
        nvonnxparser_10
        cuda
        cublas
        cudart
        cudnn
)

#add_library(${PROJECT_NAME} SHARED
#        src/base_memory.cpp
#        src/detect_postprocess.cu
#        src/infer.cpp
#        src/logger.cpp
#        src/obb_postprocess.cu
#        src/preprocess.cu
#        src/timer.cpp
#        src/yolo.cpp
#        workspace/export/export_multiple.cpp
#        workspace/export/export_multiple_cpm.cpp
#        workspace/export/export_single.cpp
#        workspace/export/export_single_cpm.cpp
#)

add_executable(${PROJECT_NAME}
        src/base_memory.cpp
        src/detect_postprocess.cu
        src/infer.cpp
        src/logger.cpp
        src/obb_postprocess.cu
        src/preprocess.cu
        src/timer.cpp
        src/yolo.cpp
        workspace/example/main.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} ${LIBS})