using System.Runtime.InteropServices;
using ConsoleApp1;
using OpenCvSharp;

class OneTest_new
{
    [DllImport(Config.YOLODLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern bool TENSORRT_INIT_ASYNC_NEW(string engineFile, float confidence, float nms);

    [DllImport(Config.YOLODLL, CallingConvention = CallingConvention.Cdecl)]
    private static extern void TENSORRT_INFER_ASYNC_NEW(IntPtr image, out IntPtr result, out int size);

    [DllImport(Config.YOLODLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern void TENSORRT_STOP_NEW();

    public static List<Box> TENSORRT_INFER_WRAPPER(IntPtr image)
    {
        TENSORRT_INFER_ASYNC_NEW(image, out IntPtr result, out int size);
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
        
        bool ok = TENSORRT_INIT_ASYNC_NEW(Config.MODEL, Config.CONFIDENCE, Config.NMS);
        if (!ok) return;
    
        Mat imRead = Cv2.ImRead(Config.IMAGE_SRC);
    
        // while (true)
        // {
            List<Box> boxes = TENSORRT_INFER_WRAPPER(imRead.CvPtr);
            foreach (var box in boxes)
            {
                Console.WriteLine($"Box: left={box.left}, top={box.top}, right={box.right}, bottom={box.bottom}, confidence={box.confidence}, class_label={box.class_label}");
            }
        // }
    
        TENSORRT_STOP_NEW();
    
    }
    
}