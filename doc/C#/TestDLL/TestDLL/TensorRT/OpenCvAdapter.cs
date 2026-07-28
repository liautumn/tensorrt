using System;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace Yolo26Net;

internal static class OpenCvAdapter
{
    internal static BgrImage ToBgrImage(Mat image)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Empty())
        {
            throw new ArgumentException("图片为空。", nameof(image));
        }

        if (image.Type() != MatType.CV_8UC3)
        {
            throw new ArgumentException("图片必须是 BGR8 三通道格式。", nameof(image));
        }

        int rowBytes = checked(image.Width * 3);
        byte[] data = new byte[checked(rowBytes * image.Height)];
        for (int row = 0; row < image.Height; ++row)
        {
            Marshal.Copy(image.Ptr(row), data, row * rowBytes, rowBytes);
        }

        return new BgrImage(data, image.Width, image.Height, rowBytes);
    }
}
