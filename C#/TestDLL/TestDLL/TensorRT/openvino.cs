using System.Runtime.InteropServices;
using ConsoleApp1;
using OpenCvSharp;

class Openvino
{
    [DllImport(Config.OPENVINO_DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern void OPENVINO_INIT(string engineFile, float confidence, float nms);

    [DllImport(Config.OPENVINO_DLL, CallingConvention = CallingConvention.Cdecl)]
    private static extern void OPENVINO_INFER(IntPtr image, out IntPtr result, out int size);

    public static List<Box2> OPENVINO_INFER_WRAPPER(IntPtr image)
    {
        OPENVINO_INFER(image, out IntPtr result, out int size);
        List<Box2> boxes = new List<Box2>(size);
        for (int i = 0; i < size; i++)
        {
            IntPtr boxPtr = IntPtr.Add(result, i * Marshal.SizeOf(typeof(Box2)));
            boxes.Add(Marshal.PtrToStructure<Box2>(boxPtr));
        }

        return boxes;
    }

    // static void Main()
    // {
    //     OPENVINO_INIT(Config.OPENVINO_MODEL, Config.CONFIDENCE, Config.NMS);
    //
    //     Mat imRead = Cv2.ImRead(Config.IMAGE_SRC);
    //     List<Box2> boxes = OPENVINO_INFER_WRAPPER(imRead.CvPtr);
    //     foreach (var box in boxes)
    //     {
    //         Console.WriteLine(
    //             $"Box: x={box.x}, y={box.y}, w={box.w}, h={box.h}, confidence={box.confidence}, class_label={box.class_label}");
    //     }
    // }
}