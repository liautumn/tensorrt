using System;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace ConsoleApp1
{
    public class Multiple
    {
        [DllImport(Config.Yolodll, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool
            TENSORRT_Multiple_INIT(string engineFile, float confidence, float nms, int maxBatch);

        [DllImport(Config.Yolodll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TENSORRT_Multiple_INFER(IntPtr[] images, int size, out IntPtr result,
            out IntPtr resultSizes);

        public static Box[][] TENSORRT_Multiple_INFER_WRAPPER(IntPtr[] images, int imgSize)
        {
            // 调用 C++ 函数
            TENSORRT_Multiple_INFER(images, imgSize, out IntPtr resultPtr, out IntPtr resultSizesPtr);

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


        static void Main()
        {
            int maxBatch = 12;

            bool initSuccess = TENSORRT_Multiple_INIT(Config.Model2, Config.Confidence, Config.Nms, maxBatch);
            if (initSuccess)
            {
                IntPtr[] imagesPtr = new IntPtr[maxBatch];
                for (int i = 0; i < maxBatch; i++)
                {
                    var mat = Cv2.ImRead(Config.ImageSrc);
                    imagesPtr[i] = mat.CvPtr;
                }

                // 调用推理函数
                Box[][] results = TENSORRT_Multiple_INFER_WRAPPER(imagesPtr, maxBatch);
                // 处理结果
                foreach (var boxes in results)
                {
                    Console.WriteLine(boxes.Length);
                    foreach (var box in boxes)
                    {
                        Console.WriteLine(
                            $"Box: left={box.left}, top={box.top}, right={box.right}, bottom={box.bottom}, confidence={box.confidence}, class_label={box.class_label}");
                    }
                }
            }
        }
    }
}