namespace ConsoleApp1;

using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
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
    public const string YOLODLL = @"D:\dev\code\CLion\tensorrt\cmake-build-release\yolo.dll";
    public const string MODEL = @"D:\dev\code\CLion\tensorrt\workspace\model\engine\best.engine";
    public const string IMAGE_SRC = @"D:\dev\code\CLion\tensorrt\workspace\images\bl.jpg";
    public const float CONFIDENCE = (float)0.45;
    public const float NMS = (float)0.5;
}