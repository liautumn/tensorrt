#ifndef PREPROCESS_CUH
#define PREPROCESS_CUH
#include <cstdint>
#include <tuple>

using namespace std;

enum class NormType : int {
    None = 0, MeanStd = 1, AlphaBeta = 2
};

enum class ChannelType : int {
    None = 0, SwapRB = 1
};

struct AffineMatrix
{
    float i2d[6]; // image to dst(network), 2x3 matrix
    float d2i[6]; // dst to image, 2x3 matrix

    void compute(const tuple<int, int>& from, const tuple<int, int>& to)
    {
        float scale_x = get<0>(to) / static_cast<float>(get<0>(from));
        float scale_y = get<1>(to) / static_cast<float>(get<1>(from));
        float scale = min(scale_x, scale_y);
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};

/* ��һ������������֧�־�ֵ��׼�alpha beta����swap RB */
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
