cmake_minimum_required(VERSION 3.22)

# 未显式指定时，为常见的 TensorRT 10 GPU 设置默认架构；可通过
# -DCMAKE_CUDA_ARCHITECTURES=86 等参数覆盖。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90" CACHE STRING
      "需要生成 CUDA 代码的 GPU 架构")
endif()

project(yolo26 VERSION 1.0.0 LANGUAGES CXX CUDA)

# 核心库和 CUDA 预处理统一使用 C++17。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(YOLO26_BUILD_EXAMPLE "构建由 config.h 驱动的 OpenCV 示例" ON)

# TensorRT 路径既可通过 -DTENSORRT_ROOT=... 传入，也可使用同名环境变量。
set(TENSORRT_ROOT "" CACHE PATH "TensorRT 10 安装目录")
if(NOT TENSORRT_ROOT)
  if(DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
  elseif(DEFINED ENV{TENSORRT_PATH})
    set(TENSORRT_ROOT "$ENV{TENSORRT_PATH}")
  endif()
endif()

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS "${TENSORRT_ROOT}"
  PATH_SUFFIXES include
  DOC "TensorRT 头文件目录"
)
find_library(TENSORRT_NVINFER_LIBRARY
  NAMES nvinfer nvinfer_10 nvinfer_10.lib
  HINTS "${TENSORRT_ROOT}"
  PATH_SUFFIXES lib lib64 lib/x64
  DOC "TensorRT nvinfer 库"
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY)
  message(FATAL_ERROR
    "未找到 TensorRT。请设置 TENSORRT_ROOT，目录内应包含 include/NvInfer.h 和 nvinfer 库。")
endif()

# 当前业务代码使用 TensorRT 10 的 I/O Tensor API，配置阶段即拒绝旧版本。
set(_tensorrt_version_header "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
if(NOT EXISTS "${_tensorrt_version_header}")
  message(FATAL_ERROR "TensorRT 安装缺少 NvInferVersion.h：${TENSORRT_INCLUDE_DIR}")
endif()
file(STRINGS "${_tensorrt_version_header}" _tensorrt_major_line
     REGEX "^#define[ \t]+NV_TENSORRT_MAJOR[ \t]+[0-9]+" LIMIT_COUNT 1)
if(NOT _tensorrt_major_line MATCHES "NV_TENSORRT_MAJOR[ \t]+10([^0-9]|$)")
  message(FATAL_ERROR "本项目仅支持 TensorRT 10，当前版本信息：${_tensorrt_major_line}")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# 统一封装 TensorRT 库，避免业务目标直接依赖平台特定的 .so/.lib 名称。
add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
set_target_properties(TensorRT::nvinfer PROPERTIES
  IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
)

# src 只收集核心 cpp/cu，C ABI 与 CPM 导出实现单独放在 workspace/export。
file(GLOB YOLO26_CORE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)
set(YOLO26_EXPORT_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/workspace/export" CACHE PATH
    "C ABI 与 CPM 导出实现目录")
file(GLOB YOLO26_EXPORT_SOURCES CONFIGURE_DEPENDS
  "${YOLO26_EXPORT_SOURCE_DIR}/*.cpp"
)
file(GLOB YOLO26_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
)

# 动态库包含核心业务和全部导出接口；示例 main 不会进入动态库。
# Linux 生成 libyolo26.so，Windows 生成 yolo26.dll。
add_library(yolo26 SHARED
  ${YOLO26_CORE_SOURCES}
  ${YOLO26_EXPORT_SOURCES}
  ${YOLO26_HEADERS}
)
target_include_directories(yolo26
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(yolo26 PRIVATE YOLO26_BUILD_DLL)
target_link_libraries(yolo26
  PRIVATE
    TensorRT::nvinfer
    CUDA::cudart
    Threads::Threads
)

# Linux 动态库、Windows 动态库和导入库统一输出到 workspace/export。
# 使用生成器表达式可避免 Visual Studio 等多配置生成器追加 Debug/Release 子目录。
set(YOLO26_BINARY_OUTPUT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/workspace/export")
set_target_properties(yolo26 PROPERTIES
  OUTPUT_NAME yolo26
  RUNTIME_OUTPUT_DIRECTORY "$<1:${YOLO26_BINARY_OUTPUT_DIR}>"
  LIBRARY_OUTPUT_DIRECTORY "$<1:${YOLO26_BINARY_OUTPUT_DIR}>"
  ARCHIVE_OUTPUT_DIRECTORY "$<1:${YOLO26_BINARY_OUTPUT_DIR}>"
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# OpenCV 只负责示例中的图片读写和画框，不会链接到 yolo26 动态库。
if(YOLO26_BUILD_EXAMPLE)
  if(NOT OpenCV_DIR AND DEFINED ENV{OPENCV_PATH})
    set(OpenCV_DIR "$ENV{OPENCV_PATH}" CACHE PATH "OpenCV CMake 配置目录")
  endif()
  find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

  add_executable(yolo26_example workspace/example/main.cpp)
  target_include_directories(yolo26_example PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(yolo26_example PRIVATE yolo26 ${OpenCV_LIBS})
endif()

# 安装动态库、导入库和公开头文件，便于 Linux/C++ 或 Windows/C# 项目部署。
include(GNUInstallDirs)
install(TARGETS yolo26
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)
