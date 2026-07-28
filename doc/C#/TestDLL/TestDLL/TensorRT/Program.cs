using System;
using System.Collections.Generic;
using System.IO;
using OpenCvSharp;

namespace Yolo26Net;

internal static class Program
{
    private static void Main()
    {
        try
        {
            Run();
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception.Message);
            Environment.ExitCode = 1;
        }
    }

    private static void Run()
    {
        ValidateConfig();
        Directory.CreateDirectory(Config.OutputDirectory);

        List<Mat> sourceImages = new(Config.ImagePaths.Length);
        try
        {
            BgrImage[] inputs = new BgrImage[Config.ImagePaths.Length];
            for (int i = 0; i < Config.ImagePaths.Length; ++i)
            {
                Mat image = Cv2.ImRead(Config.ImagePaths[i], ImreadModes.Color);
                if (image.Empty())
                {
                    image.Dispose();
                    throw new FileNotFoundException("无法读取图片。", Config.ImagePaths[i]);
                }

                sourceImages.Add(image);
                inputs[i] = OpenCvAdapter.ToBgrImage(image);
            }

            using Yolo26Detector detector = new(
                Config.EnginePath, Config.ConfidenceThreshold, Config.GpuDevice);

            if (inputs.Length == 1)
            {
                ImagePrediction prediction = detector.Predict(inputs[0]);
                PrintTiming(prediction.Timing, 1);
                SaveResult(sourceImages[0], prediction.Detections, Config.ImagePaths[0]);
                return;
            }

            BatchPrediction batch = detector.Predict(inputs);
            PrintTiming(batch.Timing, inputs.Length);
            for (int i = 0; i < inputs.Length; ++i)
            {
                SaveResult(sourceImages[i], batch.Detections[i], Config.ImagePaths[i]);
            }
        }
        finally
        {
            foreach (Mat image in sourceImages)
            {
                image.Dispose();
            }
        }
    }

    private static void SaveResult(Mat image, Detection[] detections, string inputPath)
    {
        foreach (Detection detection in detections)
        {
            Point topLeft = new(
                Math.Clamp((int)detection.Left, 0, image.Width - 1),
                Math.Clamp((int)detection.Top, 0, image.Height - 1));
            Point bottomRight = new(
                Math.Clamp((int)detection.Right, 0, image.Width - 1),
                Math.Clamp((int)detection.Bottom, 0, image.Height - 1));

            Cv2.Rectangle(image, topLeft, bottomRight, Scalar.LimeGreen, 2);
            string label = $"{Config.GetClassName(detection.ClassId)} {detection.Confidence:F2}";
            Point labelPosition = new(topLeft.X, Math.Max(20, topLeft.Y));
            Cv2.PutText(image, label, labelPosition, HersheyFonts.HersheySimplex,
                0.6, Scalar.LimeGreen, 2);

            Console.WriteLine(
                $"{Path.GetFileName(inputPath)}: [{detection.Left:F1}, {detection.Top:F1}, " +
                $"{detection.Right:F1}, {detection.Bottom:F1}] " +
                $"置信度={detection.Confidence:F3}, 类别={Config.GetClassName(detection.ClassId)}");
        }

        string extension = Path.GetExtension(inputPath);
        if (string.IsNullOrWhiteSpace(extension))
        {
            extension = ".jpg";
        }

        string outputPath = Path.Combine(
            Config.OutputDirectory,
            $"{Path.GetFileNameWithoutExtension(inputPath)}_yolo26{extension}");
        Cv2.ImWrite(outputPath, image);
        Console.WriteLine($"检测结果：{outputPath}");
    }

    private static void PrintTiming(InferenceTiming timing, int imageCount)
    {
        Console.WriteLine(
            $"N={imageCount}, 预处理={timing.PreprocessMs:F3} ms, " +
            $"推理={timing.InferenceMs:F3} ms, 后处理={timing.PostprocessMs:F3} ms, " +
            $"总耗时={timing.TotalMs:F3} ms");
    }

    private static void ValidateConfig()
    {
        if (!File.Exists(Config.NativeLibraryPath))
        {
            throw new FileNotFoundException(
                "请先构建 YOLO26 动态库，或在 Config.cs 中设置有效的 NativeLibraryPath。",
                Config.NativeLibraryPath);
        }

        if (!File.Exists(Config.EnginePath))
        {
            throw new FileNotFoundException("请在 Config.cs 中设置有效的 EnginePath。", Config.EnginePath);
        }

        if (Config.ImagePaths.Length == 0)
        {
            throw new InvalidOperationException("请在 Config.cs 中至少配置一张图片。");
        }
    }
}
