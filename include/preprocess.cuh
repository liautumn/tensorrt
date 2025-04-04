#ifndef PREPROCESS_CUH
#define PREPROCESS_CUH
#include "infer.h"

enum class NormType : int {
    None = 0, MeanStd = 1, AlphaBeta = 2
};

enum class ChannelType : int {
    None = 0, SwapRB = 1
};

/* 归一化操作，可以支持均值标准差，alpha beta，和swap RB */
struct Norm {
    float mean[3];
    float std[3];
    float alpha, beta;
    NormType type = NormType::None;
    ChannelType channel_type = ChannelType::None;

    // out = (x * alpha - mean) / std
    static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f,
                         ChannelType channel_type = ChannelType::None);

    // out = x * alpha + beta
    static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None);

    // None
    static Norm None();
};

void warp_affine_bilinear_and_normalize_plane(uint8_t *src, int src_line_size, int src_width,
                                                     int src_height, float *dst, int dst_width,
                                                     int dst_height, float *matrix_2_3,
                                                     uint8_t const_value, const Norm &norm,
                                                     cudaStream_t stream);

#endif //PREPROCESS_CUH
