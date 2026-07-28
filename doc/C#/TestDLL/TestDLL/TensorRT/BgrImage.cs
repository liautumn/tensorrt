using System;

namespace Yolo26Net;

public sealed class BgrImage
{
    public BgrImage(byte[] data, int width, int height, int stride = 0)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "图片宽高必须大于 0。");
        }

        int rowBytes = checked(width * 3);
        stride = stride == 0 ? rowBytes : stride;
        if (stride < rowBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(stride), "BGR 行跨度不能小于 width * 3。");
        }

        int requiredBytes = checked((height - 1) * stride + rowBytes);
        if (data.Length < requiredBytes)
        {
            throw new ArgumentException("BGR 数据长度不足。", nameof(data));
        }

        Data = data;
        Width = width;
        Height = height;
        Stride = stride;
    }

    public byte[] Data { get; }
    public int Width { get; }
    public int Height { get; }
    public int Stride { get; }
}
