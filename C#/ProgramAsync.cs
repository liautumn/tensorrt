using System.Runtime.InteropServices;
using ConsoleApp1;
using OpenCvSharp;


class ProgramAsync
{
    [DllImport(Config.YOLODLL, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
    public static extern bool TensorRT_INIT_ASYNC(
        [MarshalAs(UnmanagedType.LPStr)] string engineFile,
        float confidence,
        float nms);

    [DllImport(Config.YOLODLL, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr TensorRT_INFER_ASYNC(
        IntPtr image,
        ref int width,
        ref int height,
        out int size);


    [DllImport(Config.YOLODLL, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
    private static extern void FreeMemory_ASYNC(IntPtr ptr);

    // 辅助函数将 IntPtr 转换为 List<Box>
    public static List<Box> TensorRT_INFER_WRAPPER(IntPtr image, int width,
        int height)
    {
        IntPtr resultPtr = TensorRT_INFER_ASYNC(image, ref width, ref height, out int size);
        List<Box> boxes = new List<Box>(size);
        for (int i = 0; i < size; i++)
        {
            IntPtr boxPtr = IntPtr.Add(resultPtr, i * Marshal.SizeOf(typeof(Box)));
            boxes.Add(Marshal.PtrToStructure<Box>(boxPtr));
        }

        // 释放C++中分配的内存
        FreeMemory_ASYNC(resultPtr);
        return boxes;
    }

    static byte[] ReadImageToBytes(string imagePath)
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

    static void Main1()
    {
        bool ok = TensorRT_INIT_ASYNC(Config.MODEL, Config.CONFIDENCE, Config.NMS);
        if (!ok) return;

        byte[] bytes = ReadImageToBytes(Config.IMAGE_SRC);
        Mat imRead = Cv2.ImDecode(bytes, ImreadModes.Color);

        while (true)
        {
            List<Box> boxes = TensorRT_INFER_WRAPPER(imRead.Data, imRead.Cols, imRead.Rows);
            // foreach (var box in boxes)
            // {
            //     Console.WriteLine(
            //         $"Box: left={box.left}, top={box.top}, right={box.right}, bottom={box.bottom}, confidence={box.confidence}, class_label={box.class_label}");
            // }
        }
    }
}