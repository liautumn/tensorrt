using System;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace ConsoleApp1
{
    public class BatchTest
    {
        [DllImport(Config.YOLODLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool initBatchAsync(string engine_file, float confidence, float nms, int max_batch);

        [DllImport(Config.YOLODLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void inferBatchAsync(IntPtr[] images, int size, out IntPtr result,
            out IntPtr resultSizes);

        public static Box[][] InferBatchAsync(IntPtr[] images, int imgSize)
        {
            // 调用 C++ 函数
            inferBatchAsync(images, imgSize, out IntPtr resultPtr, out IntPtr resultSizesPtr);

            // 分配内存以保存结果大小
            int[] resultSizes = new int[imgSize];
            Marshal.Copy(resultSizesPtr, resultSizes, 0, imgSize);

            // 分配内存以保存结果
            Box[][] result = new Box[imgSize][];
            for (int i = 0; i < imgSize; ++i)
            {
                IntPtr boxPtr = Marshal.ReadIntPtr(resultPtr, i * IntPtr.Size);
                int boxCount = resultSizes[i];
                Box[] boxes = new Box[boxCount];
                for (int j = 0; j < boxCount; ++j)
                {
                    boxes[j] = Marshal.PtrToStructure<Box>(boxPtr + j * Marshal.SizeOf<Box>());
                }

                result[i] = boxes;
            }

            // 在适当的时候释放非托管内存
            Marshal.FreeCoTaskMem(resultPtr);
            Marshal.FreeCoTaskMem(resultSizesPtr);

            return result;
        }


        // static void Main()
        // {
        //     int maxBatch = 12;
        //
        //     bool initSuccess = initBatchAsync(Config.MODEL, Config.CONFIDENCE, Config.NMS, maxBatch);
        //     if (initSuccess)
        //     {
        //         int size = 6;
        //         IntPtr[] imagesPtr = new IntPtr[size];
        //         for (int i = 0; i < size; i++)
        //         {
        //             byte[] bytes = Utils.ReadImageToBytes(Config.IMAGE_SRC);
        //             Mat imRead = Cv2.ImDecode(bytes, ImreadModes.Color);
        //             imagesPtr[i] = imRead.CvPtr;
        //         }
        //
        //         while (true)
        //         {
        //             // 调用推理函数
        //             Box[][] results = InferBatchAsync(imagesPtr, size);
        //             // 处理结果
        //             foreach (var boxes in results)
        //             {
        //                 Console.WriteLine(boxes.Length);
        //                 foreach (var box in boxes)
        //                 {
        //                     Console.WriteLine(
        //                         $"Box: left={box.left}, top={box.top}, right={box.right}, bottom={box.bottom}, confidence={box.confidence}, class_label={box.class_label}");
        //                 }
        //             }
        //         }
        //     }
        // }
    }
}