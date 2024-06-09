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
    public const string YOLODLL = @"D:\autumn\Documents\JetBrainsProjects\CLion\tensorrt\cmake-build-release\yolo.dll";

    public const string MODEL =
        @"D:\autumn\Documents\JetBrainsProjects\CLion\tensorrt\workspace\model\engine\yolov8s.transd.engine";

    public const string IMAGE_SRC =
        @"D:\autumn\Documents\JetBrainsProjects\CLion\tensorrt\workspace\images\car.jpg";

    public const float CONFIDENCE = (float)0.25;
    public const float NMS = (float)0.7;
}

public class Utils
{
    public static byte[] ReadImageToBytes(string imagePath)
    {
        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException("Image file not found.", imagePath);
        }

        byte[] imageBytes;
        try
        {
            imageBytes = File.ReadAllBytes(imagePath);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred while reading the image: {ex.Message}");
            throw;
        }

        return imageBytes;
    }
}