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
        // if (!ok) return;

        // Mat imRead = Cv2.ImRead(Config.IMAGE_SRC);
        // List<Box> boxes = TENSORRT_INFER_WRAPPER(imRead.CvPtr);
        //
        // // 绘制检测框和标签
        // foreach (var box in boxes)
        // {
        //     var p1 = new Point(box.left, box.top);
        //     var p2 = new Point(box.right, box.bottom);
        //     Cv2.Rectangle(imRead, p1, p2, Scalar.Blue, 3);
        //     var labelPosition = new Point(box.left + 5, box.top - 5); // 向右和向上偏移5像素
        //     Cv2.PutText(imRead, Config.classList[box.class_label],
        //         labelPosition,
        //         HersheyFonts.HersheySimplex, 1, Scalar.Blue, 3);
        //     Console.WriteLine(
        //         $"Box: left={box.left}, top={box.top}, right={box.right}, bottom={box.bottom}, confidence={box.confidence}, " +
        //         $"class_label={Config.classList[box.class_label]}");
        // }
        //
        // // 设置窗口大小
        // string windowName = "1";
        // Cv2.NamedWindow(windowName, WindowFlags.Normal); // 窗口可调整大小
        // Cv2.ResizeWindow(windowName, 800, 600); // 设置窗口大小为 800x600
        //
        // // 显示图像
        // Cv2.ImShow(windowName, imRead);
        // Cv2.WaitKey();
        //
        // TENSORRT_STOP_NEW();
    }
}