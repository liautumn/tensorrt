namespace ConsoleApp1;

using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public struct Box
{
    public float left;
    public float top;
    public float right;
    public float bottom;
    public float confidence;
    public int class_label;
}

public class Config
{
    public const string YOLODLL = @"D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\cmake-build-release\yolo.dll";

    public const string MODEL = @"D:\autumn\Documents\WeChat Files\wxid_1w2acnt5bx6s22\FileStorage\File\2024-07\best_4.0_4090.engine";

    public const string IMAGE_SRC = @"D:\autumn\Pictures\20240729143358.jpg";

    public const float CONFIDENCE = (float)0.25;
    public const float NMS = (float)0.4;
}