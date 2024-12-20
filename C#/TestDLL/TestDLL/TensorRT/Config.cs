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
    public const string OpenvinoDll = @"D:\autumn\Documents\JetBrainsProjects\CLionProjects\OpenVINO\yolov8\cmake-build-release\openvino_yolo.dll"; 
    public const string OpenvinoModel = @"D:\autumn\Documents\JetBrainsProjects\CLionProjects\OpenVINO\yolov8\yolo\best_openvino_model\best.xml";

    public const string Yolodll = @"D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\cmake-build-release\tensorrt.dll";
    public const string Model = @"D:\\autumn\\Documents\\JetBrainsProjects\\CLionProjects\\tensorrt\\workspace\\model\\engine\\best.transd.engine";
    public const string Model2 = @"D:\\autumn\\Documents\\JetBrainsProjects\\CLionProjects\\tensorrt\\workspace\\model\\engine\\best.transd.engine";

    public const string ImageSrc = @"F:\FlawImages2\b8d95714-2b8c-4959-92fa-959ce86cc50f.jpeg";

    public const float Confidence = (float)0.2;
    public const float Nms = (float)0.2;
    
    public static string[] ClassList = {
        "BB",
        "ZH",
        "ZK",
        "JK",
        "ZZ",
        "GS",
        "ZW",
        "DJ",
        "PD",
        "CS",
        "DW",
        "HN",
        "YW",
        "FH",
        "LZ",
        "SYQ",
        "BQ",
        "DPD",
        "MD",
        "CH",
        "SD",
        "SZ",
        "ZS"
    };


}