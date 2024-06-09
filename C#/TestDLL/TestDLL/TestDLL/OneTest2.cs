using System.Runtime.InteropServices;
using ConsoleApp1;
using OpenCvSharp;

class OneTest2
{
    [DllImport(Config.YOLODLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern bool TensorRT_INIT_ASYNC2(string engine_file, float confidence, float nms);

    [DllImport(Config.YOLODLL, CallingConvention = CallingConvention.Cdecl)]
    private static extern void TensorRT_INFER_ASYNC2(IntPtr image, out IntPtr result, out int size);


    // 辅助函数将 IntPtr 转换为 List<Box>
    public static List<Box> TensorRT_INFER_WRAPPER(IntPtr image)
    {
        TensorRT_INFER_ASYNC2(image, out IntPtr result, out int size);
        List<Box> boxes = new List<Box>(size);
        for (int i = 0; i < size; i++)
        {
            IntPtr boxPtr = IntPtr.Add(result, i * Marshal.SizeOf(typeof(Box)));
            boxes.Add(Marshal.PtrToStructure<Box>(boxPtr));
        }
        return boxes;
    }

    static void Main()
    {
        bool ok = TensorRT_INIT_ASYNC2(Config.MODEL, Config.CONFIDENCE, Config.NMS);
        if (!ok)
        {
            return;
        }

        byte[] bytes = Utils.ReadImageToBytes(Config.IMAGE_SRC);
        Mat imRead = Cv2.ImDecode(bytes, ImreadModes.Color);

        while (true)
        {
            List<Box> boxes = TensorRT_INFER_WRAPPER(imRead.CvPtr);
            foreach (var box in boxes)
            {
                Console.WriteLine(
                    $"Box: left={box.left}, top={box.top}, right={box.right}, bottom={box.bottom}, confidence={box.confidence}, class_label={box.class_label}");
            }
        }
    }
}