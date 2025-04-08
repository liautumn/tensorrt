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
#        src/BaseMemory.cpp
#        src/Infer.cpp
#        src/Logger.cpp
#        src/Postprocess.cu
#        src/Preprocess.cu
#        src/Timer.cpp
#        src/Yolo.cpp
#        src/export/ExportMultiple.cpp
#        src/export/ExportMultipleCpm.cpp
#        src/export/ExportSingle.cpp
#        src/export/ExportSingleCpm.cpp
#)

add_executable(${PROJECT_NAME}
        src/BaseMemory.cpp
        src/Infer.cpp
        src/Logger.cpp
        src/Postprocess.cu
        src/Preprocess.cu
        src/Timer.cpp
        src/Yolo.cpp
        src/example/Config.h
        src/example/Main.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} ${LIBS})