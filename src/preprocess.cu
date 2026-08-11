// SPDX-FileCopyrightText: 2026 liqiuzhuang and contributors
// SPDX-License-Identifier: GPL-3.0-only

// 引入 CUDA letterbox 预处理接口和相关项目类型。
#include "preprocess.h"

// CUDA Runtime 完整接口：提供 kernel 启动、pinned memory 和异步内存复制。
#include <cuda_runtime.h>

// std::min 用于计算保持宽高比的统一缩放比例。
#include <algorithm>
// std::size_t 用于计算 workspace 字节数和图片偏移。
#include <cstddef>
// floorf 用于寻找双线性采样低坐标，并按 OpenCV 风格对插值结果取整。
#include <cmath>
// std::memcpy 用于把普通 cv::Mat 数据复制到 pinned memory。
#include <cstring>
#include <utility>
// std::vector 用于保存批次内每张图片在 workspace 中的偏移。
#include <vector>

// 集中使用 Assertf、checkRuntime 和 kernel 启动检查。
#include "validator.h"

namespace
{

// letterbox 区域使用与 yolov8+Trt10 分支相同的灰色像素值；归一化后为 114/255。
constexpr unsigned char kLetterboxValue = 114;
// 把矩阵表后的首张图起点对齐到 256 字节，矩阵区和图片区使用清晰、稳定的边界。
constexpr std::size_t kWorkspaceAlignment = 256;

// 把 value 向上取整到 alignment 的整数倍；调用方保证 alignment 大于 0。
// 返回值仍以字节为单位，用作 workspace 中下一段数据的起点。
[[nodiscard]] constexpr std::size_t alignUp(
    std::size_t const value,
    std::size_t const alignment) noexcept
{
    // 先补 alignment-1 再做整数除法，可得到不小于 value 的最小对齐值。
    return (value + alignment - 1) / alignment * alignment;
}

// 为一张原图计算保持宽高比的 letterbox 仿射矩阵及其逆矩阵。
// sourceWidth/sourceHeight 是原图尺寸，destinationWidth/destinationHeight 是模型输入尺寸。
AffineMatrix makeAffineMatrix(int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight)
{
    // 分别计算把原图宽、高完整缩放到目标宽、高所需的比例。
    float const scaleX = static_cast<float>(destinationWidth) / sourceWidth;
    float const scaleY = static_cast<float>(destinationHeight) / sourceHeight;
    // 取较小比例能让整张图落入模型输入，剩余区域填充 114 而不是裁剪原图。
    float const scale = std::min(scaleX, scaleY);

    // 两组 2x3 矩阵都按行保存为 [m0,m1,m2,m3,m4,m5]。
    AffineMatrix matrix;
    // x 方向只做统一缩放，不引入旋转或错切，因此交叉项为 0。
    matrix.i2d[0] = scale;
    matrix.i2d[1] = 0.0F;
    // x 平移使缩放后的图像居中；scale*0.5-0.5 是目标分支使用的像素中心修正。
    matrix.i2d[2] = -scale * sourceWidth * 0.5F + destinationWidth * 0.5F + scale * 0.5F - 0.5F;
    matrix.i2d[3] = 0.0F;
    // y 方向使用同一个 scale，从而保持原图宽高比。
    matrix.i2d[4] = scale;
    // y 平移同样居中缩放后的图像，并应用相同的半像素修正。
    matrix.i2d[5] = -scale * sourceHeight * 0.5F + destinationHeight * 0.5F + scale * 0.5F - 0.5F;

    // 计算 i2d 左侧 2x2 线性部分的行列式，并取倒数用于构造逆矩阵。
    double determinant
        = static_cast<double>(matrix.i2d[0]) * matrix.i2d[4]
        - static_cast<double>(matrix.i2d[1]) * matrix.i2d[3];
    // 有效图片尺寸一定得到非零 scale；零值分支只是防止异常输入导致除零。
    determinant = determinant != 0.0 ? 1.0 / determinant : 0.0;

    // a11~a22 是逆矩阵的 2x2 线性部分；中间计算用 double 减少求逆误差。
    double const a11 = matrix.i2d[4] * determinant;
    double const a12 = -matrix.i2d[1] * determinant;
    double const a21 = -matrix.i2d[3] * determinant;
    double const a22 = matrix.i2d[0] * determinant;

    // d2i 线性部分最终写回 float，和 CUDA kernel 及目标分支的数据类型保持一致。
    matrix.d2i[0] = static_cast<float>(a11);
    matrix.d2i[1] = static_cast<float>(a12);
    // 逆矩阵的平移部分满足 b_inverse = -A_inverse * b_original。
    matrix.d2i[2] = static_cast<float>(-a11 * matrix.i2d[2] - a12 * matrix.i2d[5]);
    matrix.d2i[3] = static_cast<float>(a21);
    matrix.d2i[4] = static_cast<float>(a22);
    matrix.d2i[5] = static_cast<float>(-a21 * matrix.i2d[2] - a22 * matrix.i2d[5]);
    // i2d 描述正向 letterbox；同一份 d2i 同时交给 kernel 和后处理使用。
    return matrix;
}

// 确保 Model 中的 pinned CPU workspace 和 GPU workspace 都能容纳当前批次。
// 两块内存始终使用相同容量与布局，并在后续批次复用，避免每轮 cudaMalloc/cudaFree。
void ensureWorkspaceCapacity(Model& model, std::size_t requiredBytes)
{
    // workspace 属于传入的 Model；调用方按顺序复用同一实例的 workspace。
    // 已有容量足够时直接复用，不进行任何分配或释放。
    if (model.preprocessCapacity >= requiredBytes)
    {
        return;
    }

    // 从这一刻起旧容量不再可复用；即使释放中途抛异常，下次也不会错误地提前返回。
    model.preprocessCapacity = 0;
    // 先释放 device workspace；公开函数在上一轮返回前已经同步同一 stream。
    model.preprocessDevice.reset();
    // device 释放成功后再释放配对的 pinned host workspace。
    model.preprocessHost.reset();

    // cudaMallocHost 创建 page-locked 内存，使下面的 H2D 真正支持 cudaMemcpyAsync。
    CudaPinnedMemory newHost = allocateCudaPinned(requiredBytes);

    // 为同一批打包数据申请 device 端镜像 workspace。
    CudaDeviceMemory newDevice = allocateCudaDevice(requiredBytes);

    // 两次分配均成功后一次性提交地址和容量，Model 再次处于完整可用状态。
    model.preprocessHost = std::move(newHost);
    model.preprocessDevice = std::move(newDevice);
    model.preprocessCapacity = requiredBytes;
}

// 每个 CUDA thread 负责模型输入中的一个目标像素。
// kernel 用 d2i 做反向仿射采样，避免正向映射产生空洞或多个源像素竞争同一目标像素。
__global__ void letterboxKernel(
    // 当前图片在 device workspace 中紧密排列的 BGR uint8 数据。
    unsigned char const* source,
    // 原图一行的字节数；图片已打包，因此固定等于 sourceWidth * 3。
    int sourceStride,
    // 原始图片宽度，用于 x 方向采样和边界判断。
    int sourceWidth,
    // 原始图片高度，用于 y 方向采样和边界判断。
    int sourceHeight,
    // 当前 batch slot 的 FP32 NCHW 输入起点，kernel 直接写入这块 TensorRT 显存。
    float* destination,
    // 模型输入宽度，决定输出平面行跨度。
    int destinationWidth,
    // 模型输入高度，与 destinationWidth 一起决定 thread 覆盖范围。
    int destinationHeight,
    // 当前图片的 2x3 d2i，方向是模型输入坐标到原图坐标。
    float const* destinationToImage)
{
    // 由二维 block/grid 计算当前 thread 对应的模型输入像素坐标。
    int const destinationX = blockDim.x * blockIdx.x + threadIdx.x;
    int const destinationY = blockDim.y * blockIdx.y + threadIdx.y;
    // grid 会向上取整，多出的 thread 不对应有效目标像素，直接退出。
    if (destinationX >= destinationWidth || destinationY >= destinationHeight)
    {
        return;
    }

    // 用 d2i 第一行把目标整数像素反投影到原图浮点 x 坐标。
    float const sourceX
        = destinationToImage[0] * destinationX
        + destinationToImage[1] * destinationY
        + destinationToImage[2];
    // 用 d2i 第二行得到对应的原图浮点 y 坐标。
    float const sourceY
        = destinationToImage[3] * destinationX
        + destinationToImage[4] * destinationY
        + destinationToImage[5];

    // OpenCV imread 的通道顺序是 BGR；先按原顺序插值，写输出时再换成 RGB。
    float blue;
    float green;
    float red;
    // 采样中心完全落在原图外时，三个通道直接使用 letterbox 常量 114。
    if (sourceX <= -1.0F || sourceX >= sourceWidth || sourceY <= -1.0F || sourceY >= sourceHeight)
    {
        blue = kLetterboxValue;
        green = kLetterboxValue;
        red = kLetterboxValue;
    }
    else
    {
        // 双线性插值读取采样点周围的左上、右上、左下、右下四个邻居。
        int const xLow = static_cast<int>(floorf(sourceX));
        int const yLow = static_cast<int>(floorf(sourceY));
        int const xHigh = xLow + 1;
        int const yHigh = yLow + 1;

        // 小数部分表示采样点到低坐标像素的距离，据此计算四个邻居的权重。
        float const xFraction = sourceX - xLow;
        float const yFraction = sourceY - yLow;
        float const xInverse = 1.0F - xFraction;
        float const yInverse = 1.0F - yFraction;
        // weight1~4 依次对应左上、右上、左下、右下。
        float const weight1 = xInverse * yInverse;
        float const weight2 = xFraction * yInverse;
        float const weight3 = xInverse * yFraction;
        float const weight4 = xFraction * yFraction;

        // 采样点靠近图片边缘时，某些邻居可能越界；这些邻居单独用 114 参与混合。
        unsigned char const border[3]{kLetterboxValue, kLetterboxValue, kLetterboxValue};
        // 四个指针先全部指向 border，只有坐标有效时才替换成真实像素地址。
        unsigned char const* pixel1 = border;
        unsigned char const* pixel2 = border;
        unsigned char const* pixel3 = border;
        unsigned char const* pixel4 = border;

        // yLow 在外层范围判断后一定小于 sourceHeight，这里只需检查下边界。
        if (yLow >= 0)
        {
            // 左上邻居位于原图内时，按 stride 和 BGR 三通道偏移定位它。
            if (xLow >= 0)
            {
                pixel1 = source + yLow * sourceStride + xLow * 3;
            }
            // 右上邻居的 x 可能恰好等于 sourceWidth，此时继续保留 border。
            if (xHigh < sourceWidth)
            {
                pixel2 = source + yLow * sourceStride + xHigh * 3;
            }
        }
        // yHigh 在外层范围判断后一定不小于 0，这里只需检查上边界。
        if (yHigh < sourceHeight)
        {
            // 左下邻居有效时替换 pixel3。
            if (xLow >= 0)
            {
                pixel3 = source + yHigh * sourceStride + xLow * 3;
            }
            // 右下邻居同时满足 x/y 上边界时替换 pixel4。
            if (xHigh < sourceWidth)
            {
                pixel4 = source + yHigh * sourceStride + xHigh * 3;
            }
        }

        // 与目标分支/OpenCV 风格一致：插值后先 floor(value + 0.5) 到 uint8 精度。
        blue = floorf(weight1 * pixel1[0] + weight2 * pixel2[0]
            + weight3 * pixel3[0] + weight4 * pixel4[0] + 0.5F);
        green = floorf(weight1 * pixel1[1] + weight2 * pixel2[1]
            + weight3 * pixel3[1] + weight4 * pixel4[1] + 0.5F);
        red = floorf(weight1 * pixel1[2] + weight2 * pixel2[2]
            + weight3 * pixel3[2] + weight4 * pixel4[2] + 0.5F);
    }

    // 一个通道平面包含 H*W 个 float；NCHW 中 R、G、B 三个平面连续存放。
    int const area = destinationWidth * destinationHeight;
    // destinationIndex 是当前像素在单个 HxW 平面内的一维下标。
    int const destinationIndex = destinationY * destinationWidth + destinationX;
    // 把 BGR 改为 RGB，并在写入 FP32 NCHW 时执行 1/255 归一化。
    destination[destinationIndex] = red / 255.0F;
    // 第二个通道平面保存 green。
    destination[area + destinationIndex] = green / 255.0F;
    // 第三个通道平面保存 blue。
    destination[2 * area + destinationIndex] = blue / 255.0F;
}

} // namespace

// 打包并预处理当前推理批次；函数不会一次处理 images 中不属于 batch 的其他图片。
AffineMatrices preprocessBatchToGpu(
    Model& model,
    std::span<cv::Mat const> const images,
    Batch const& batch)
{
    // batch 至少包含一张图片，并且不能超出 Engine profile 允许的最大 batch。
    Assertf(batch.size > 0 && batch.size <= model.maxBatch,
        "Batch size %d is outside [1,%d]", batch.size, model.maxBatch);
    // 先分别比较 offset 和剩余数量，避免直接计算 offset+size 时发生 size_t 回绕。
    std::size_t const batchSize = static_cast<std::size_t>(batch.size);
    Assertf(batch.offset <= images.size() && batchSize <= images.size() - batch.offset,
        "Batch offset %zu and size %zu exceed %zu images", batch.offset, batchSize, images.size());

    // matrices 保留 CPU 端双向矩阵；返回后由后处理复用同一份 d2i 还原检测框。
    AffineMatrices matrices;
    // 预留当前 batch 的精确元素数，避免 push_back 过程中反复扩容。
    matrices.reserve(batchSize);
    // imageOffsets[index] 记录第 index 张图在 host/device workspace 中的字节起点。
    std::vector<std::size_t> imageOffsets(batchSize);

    // workspace 前部连续存放 batch.size 个 d2i，每个矩阵固定为 6 个 float。
    std::size_t const matrixBytes = batchSize * 6 * sizeof(float);
    // 图片从对齐地址开始，布局为 [d2i矩阵表][padding][image0][image1]...。
    std::size_t nextImageOffset = alignUp(matrixBytes, kWorkspaceAlignment);
    // 第一遍只校验图片、计算矩阵和总字节数，随后可以一次申请足够的 workspace。
    for (int index = 0; index < batch.size; ++index)
    {
        // batch.offset 把批次内下标转换为 images 中的全局下标。
        cv::Mat const& image = images[batch.offset + index];
        // kernel 固定按三通道 uint8 BGR 读取，因此拒绝空图或其他 OpenCV 类型。
        Assertf(!image.empty() && image.dims == 2
                && image.rows > 0 && image.cols > 0 && image.type() == CV_8UC3,
            "CUDA preprocessing requires a non-empty 2D CV_8UC3 image");

        // 先保存本图起点，再让下一起点越过当前图的 width*height*3 字节。
        imageOffsets[index] = nextImageOffset;
        nextImageOffset += static_cast<std::size_t>(image.cols) * image.rows * 3;
        // batch 内图片尺寸可以不同，所以每张图各自计算一组 i2d/d2i。
        matrices.push_back(makeAffineMatrix(
            image.cols, image.rows, model.inputWidth, model.inputHeight));
    }

    // 循环结束后的 nextImageOffset 就是矩阵表、对齐区和全部原图的总字节数。
    ensureWorkspaceCapacity(model, nextImageOffset);
    // 转成字节指针后才能按上面计算的 byte offset 访问两块 workspace。
    auto* hostWorkspace = static_cast<unsigned char*>(model.preprocessHost.get());
    auto* deviceWorkspace = static_cast<unsigned char*>(model.preprocessDevice.get());

    // 第二遍把每张图的 d2i 和原始 BGR 数据写入 pinned host workspace。
    for (int index = 0; index < batch.size; ++index)
    {
        // kernel 只需要“模型输入到原图”的 d2i，因此每张图复制 6 个 float。
        std::memcpy(
            hostWorkspace + static_cast<std::size_t>(index) * 6 * sizeof(float),
            matrices[index].d2i.data(),
            6 * sizeof(float));

        cv::Mat const& image = images[batch.offset + index];
        // packed 图片每行只保留有效 width*3 字节，不把 cv::Mat 的行尾 padding 带入 GPU。
        std::size_t const rowBytes = static_cast<std::size_t>(image.cols) * 3;
        // destinationRow 指向当前图片在 pinned workspace 中的第 0 行。
        unsigned char* destinationRow = hostWorkspace + imageOffsets[index];
        // 逐行复制可以兼容 ROI 等 step 大于 width*3 的非连续 cv::Mat。
        for (int row = 0; row < image.rows; ++row)
        {
            // image.ptr(row) 使用 cv::Mat 自己的 step 定位源行，目标行则保持紧密排列。
            std::memcpy(destinationRow + static_cast<std::size_t>(row) * rowBytes,
                image.ptr(row), rowBytes);
        }
    }

    // 整批矩阵和原图只发起一次异步 H2D；后续 kernel 在同一 stream 中自然等待复制完成。
    checkRuntime(cudaMemcpyAsync(deviceWorkspace, hostWorkspace, nextImageOffset,
        cudaMemcpyHostToDevice, model.stream.get()));

    // 使用目标分支相同的 32x32 thread block，每个 block 最多处理 1024 个目标像素。
    dim3 const block(32, 32);
    // grid 在宽高方向分别向上取整，kernel 内部会过滤边缘多出的 thread。
    dim3 const grid(
        (model.inputWidth + block.x - 1) / block.x,
        (model.inputHeight + block.y - 1) / block.y);
    // 每个 batch slot 固定占 3*H*W 个 float，用于计算 inputDevice 的写入偏移。
    std::size_t const inputElements
        = static_cast<std::size_t>(3) * model.inputHeight * model.inputWidth;

    // 每张图启动一次 kernel；所有 kernel 都进入 model.stream 并按批次顺序提交。
    for (int index = 0; index < batch.size; ++index)
    {
        // 再次取得原图尺寸，作为本次 kernel 的 sourceWidth/sourceHeight。
        cv::Mat const& image = images[batch.offset + index];
        // source 指向本图在 device workspace 中紧密排列的 BGR 数据。
        auto const* source = deviceWorkspace + imageOffsets[index];
        // 矩阵表从 workspace 起点开始，每张图占连续 6 个 float。
        auto const* destinationToImage
            = reinterpret_cast<float const*>(deviceWorkspace + static_cast<std::size_t>(index) * 6 * sizeof(float));
        // destination 直接指向 TensorRT 输入显存中的第 index 个 NCHW batch slot。
        auto* destination = static_cast<float*>(model.inputDevice.get())
            + static_cast<std::size_t>(index) * inputElements;

        // sourceStride 使用 packed 后的 width*3，而不是原 cv::Mat 可能更大的 step。
        checkKernel(letterboxKernel<<<grid, block, 0, model.stream.get()>>>(
            source,
            image.cols * 3,
            image.cols,
            image.rows,
            destination,
            model.inputWidth,
            model.inputHeight,
            destinationToImage));
        // checkKernel 检查启动参数等立即错误；执行期错误会在下面同步时报告。
    }

    // 当前项目按阶段同步执行：这里等待 H2D 和本批全部预处理 kernel 完成。
    // 因此 CPU chrono 的 preprocess 数值包含 GPU 工作，pinned workspace 也可在下一轮安全复用。
    checkRuntime(cudaStreamSynchronize(model.stream.get()));
    // 返回 CPU 端矩阵；其 d2i 与 kernel 使用的矩阵来自同一份数据，不会出现公式偏差。
    return matrices;
}
