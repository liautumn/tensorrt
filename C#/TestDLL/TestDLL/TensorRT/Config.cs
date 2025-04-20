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

[StructLayout(LayoutKind.Sequential)]
public struct Box2
{
    public float x;
    public float y;
    public float w;
    public float h;
    public float confidence;
    public int class_label;
}

public class Config
{
    public const string OpenvinoDll =
        @"D:\autumn\Documents\JetBrainsProjects\CLionProjects\OpenVINO\yolov8\cmake-build-release\openvino_yolo.dll";

    public const string OpenvinoModel =
        @"D:\autumn\Documents\JetBrainsProjects\CLionProjects\OpenVINO\yolov8\yolo\yolo11n_openvino_model\yolo11n.xml";
    
    
    

    public const string Yolodll =
        @"C:\ProgramData\Autumn\tensorrt.dll";

    public const string Model =
        @"D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\model\engine\best8.0.engine";

    public const string ImageSrc =
        @"D:\autumn\Desktop\1_10_1744877524.jpeg";

    public const float Confidence = (float)0.01;
    public const float Nms = (float)0.45;
    
}