using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using OpenCvSharp;

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


class ProgramAsync
{
    private const string DllName = @"D:\dev\code\CLion\tensorrt\cmake-build-release\yolo.dll";

    [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    public static extern bool TensorRT_INIT_ASYNC([MarshalAs(UnmanagedType.LPStr)] string engine_file);


    [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    private static extern int TensorRT_INFER_ASYNC(IntPtr image, out int size);


    [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    public static extern int GetListBoxData(int boxLabel, ref float left, ref float top, ref float right,
        ref float bottom, ref float confidence, ref int classLabel);

    [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    public static extern void EndGetListBoxData();


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

    // static void Main()
    // {
    //     const string engine_file =
    //         @"D:\dev\code\CLion\tensorrt\workspace\model\engine\best.engine";
    //     bool ok = TensorRT_INIT_ASYNC(engine_file);
    //     if (!ok)
    //     {
    //         return;
    //     }
    //
    //     string image_src = @"D:\dev\code\CLion\tensorrt\workspace\images\d_0000100_2024-05-31-15-25-00_c02.jpg";
    //     byte[] bytes = ReadImageToBytes(image_src);
    //
    //     Mat imRead = Cv2.ImDecode(bytes, ImreadModes.Color);
    //
    //     List<Box> boxes = TensorRT_INFER_WRAPPER(imRead.Data);
    //
    //     foreach (var box in boxes)
    //     {
    //         Console.WriteLine(
    //             $"Box: left={box.left}, top={box.top}, right={box.right}, bottom={box.bottom}, confidence={box.confidence}, class_label={box.class_label}");
    //     }
    // }


    static void Main()
    {
        const string engine_file =
            @"D:\dev\code\CLion\tensorrt\workspace\model\engine\best.engine";
        bool ok = TensorRT_INIT_ASYNC(engine_file);
        if (!ok)
        {
            return;
        }

        string image_src = @"D:\dev\code\CLion\tensorrt\workspace\images\d_0000100_2024-05-31-15-25-00_c02.jpg";
        byte[] bytes = ReadImageToBytes(image_src);

        Mat imRead = Cv2.ImDecode(bytes, ImreadModes.Color);

        int boxCount = TensorRT_INFER_ASYNC(imRead.Data, out int size);

        // int boxCount = BeiginGetListBoxData();
        if (boxCount <= 0)
        {
            Console.WriteLine("获取box列表大小失败!");
            return;
        }

        for (int i = 0; i < boxCount; ++i)
        {
            float left = 0, top = 0, right = 0, bottom = 0, confidence = 0;
            int classLabel = 0;

            if (GetListBoxData(i, ref left, ref top, ref right, ref bottom, ref confidence, ref classLabel) != 0)
            {
                Console.WriteLine($"获取box数据失败! 索引: {i}");
                EndGetListBoxData();
                return;
            }

            Console.WriteLine(
                $"Box {i}: 左: {left}, 上: {top}, 右: {right}, 下: {bottom}, 置信度: {confidence}, 类别: {classLabel}");
        }

        EndGetListBoxData();
    }
}