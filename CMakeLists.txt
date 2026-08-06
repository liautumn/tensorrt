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
    set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation directory")
endif()

# 手动选择 nvcc 时必须在 enable_language(CUDA) 前设置；命令行或缓存已有值时优先使用它。
if(WIN32 AND NOT CMAKE_CUDA_COMPILER)
    # 根据上面的 CUDA Toolkit 根目录拼出本机默认 nvcc.exe 候选路径。
    set(_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc.exe")
    # 仅在文件真实存在时写入 compiler，避免一个无效默认路径阻止 CMake 自行搜索 CUDA。
    if(EXISTS "${_CUDA_COMPILER}")
        # CACHE 让首次配置选定的 nvcc 在后续 CLion/CMake 重新加载时保持一致。
        set(CMAKE_CUDA_COMPILER "${_CUDA_COMPILER}"
            CACHE FILEPATH "CUDA compiler used to build .cu sources")
    endif()
endif()

# 启用 CUDA language，使 add_executable 中的 src/preprocess.cu 交给 nvcc 编译。
enable_language(CUDA)
# 指定 CUDA host/device 代码使用 C++17。
set(CMAKE_CUDA_STANDARD 17)
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

# 每个源文件对应 main() 中的一个学习模块。
add_executable(yolo26_trt
    # 程序入口：按执行顺序拼装下面的学习模块。
    src/main.cpp
    # 模型模块：读取 Engine、创建 context、申请和释放显存。
    src/model.cpp
    # 分批模块：根据最大 batch 生成 4、4、2 之类的批次信息。
    src/batch.cpp
    # 图片模块：从磁盘读取原始 BGR 图片。
    src/image.cpp
    # CUDA 预处理模块：letterbox、双线性插值、归一化并写入模型输入显存。
    src/preprocess.cu
    # 推理模块：设置 batch、enqueueV3 并复制输出。
    src/inference.cpp
    # CUDA Event 计时模块：测量指定 CUDA stream 上两个事件之间的耗时。
    src/timer.cpp
    # 结果模块：按 batch 下标解析、过滤并返回检测结果。
    src/result.cpp)
# 给 yolo26_trt 添加 TensorRT 和 OpenCV 头文件搜索目录。
target_include_directories(yolo26_trt PRIVATE
    # 当前项目自己的 model.h、image.h 等头文件所在目录。
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # TensorRT 的 NvInfer.h 所在目录。
    ${TENSORRT_INCLUDE_DIR}
    # OpenCV 的 opencv2 头文件所在目录。
    ${OpenCV_INCLUDE_DIRS})
# 指定最终链接时需要的动态库。
target_link_libraries(yolo26_trt PRIVATE
    # TensorRT 核心推理库 libnvinfer。
    ${TENSORRT_LIBRARY}
    # CUDA Runtime 库，提供 cudaMalloc、cudaMemcpy、CUDA stream 等函数。
    CUDA::cudart
    # OpenCV 配置导出的库集合，为 cv::Mat、imread、rectangle、putText 和 imshow 提供实现。
    ${OpenCV_LIBS})

# Windows 构建后把运行时 DLL 放到 exe 旁边，使 CLion/IDE 启动时不依赖机器 PATH。
if(WIN32)
    # 在 TensorRT 根目录的 bin 或 lib 中查找 TensorRT 11 运行时 DLL。
    find_file(TENSORRT_RUNTIME_DLL
        # 兼容带主版本后缀和不带后缀的 DLL 名称。
        NAMES nvinfer_11.dll nvinfer.dll
        # 优先使用 CMake 变量或同名环境变量指定的 TensorRT 根目录。
        HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
        # 在根目录下面的 bin、lib 子目录查找。
        PATH_SUFFIXES bin lib)
    # 如果只有导入库而没有运行时 DLL，程序仍然无法在 Windows 启动。
    if(NOT TENSORRT_RUNTIME_DLL)
        message(FATAL_ERROR "TensorRT runtime DLL not found under TENSORRT_ROOT")
    endif()

    # yolo26_trt 构建完成后执行一次文件复制命令。
    add_custom_command(TARGET yolo26_trt POST_BUILD
        # copy_if_different 只在文件内容不同时复制，避免不必要的重复写入。
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            # 第一个参数是找到的 TensorRT DLL 源路径。
            "${TENSORRT_RUNTIME_DLL}"
            # 第二个参数是生成的 yolo26_trt.exe 所在目录。
            "$<TARGET_FILE_DIR:yolo26_trt>"
        # VERBATIM 让 CMake 正确处理路径中的空格和转义字符。
        VERBATIM)

    # 官方 OpenCV Windows 包通常提供 opencv_world 这个合并后的动态库目标。
    if(TARGET opencv_world)
        # 构建完成后把 OpenCV 运行时 DLL 同样复制到 exe 目录。
        add_custom_command(TARGET yolo26_trt POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                # $<TARGET_FILE:opencv_world> 会解析为实际 OpenCV DLL 路径。
                "$<TARGET_FILE:opencv_world>"
                # 复制目标仍然是 yolo26_trt.exe 所在目录。
                "$<TARGET_FILE_DIR:yolo26_trt>"
            VERBATIM)
    endif()
endif()
