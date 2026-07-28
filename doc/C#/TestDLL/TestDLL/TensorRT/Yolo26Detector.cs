using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Yolo26Net;

public sealed class Yolo26Detector : IDisposable
{
    private readonly object syncRoot = new();
    private readonly DetectorSafeHandle detector;
    private bool disposed;

    public Yolo26Detector(string enginePath, float confidenceThreshold = 0.25F, int gpuDevice = 0)
    {
        if (string.IsNullOrWhiteSpace(enginePath))
        {
            throw new ArgumentException("Engine 路径不能为空。", nameof(enginePath));
        }

        int status = NativeMethods.CreateDetector(
            enginePath, confidenceThreshold, gpuDevice, out DetectorSafeHandle handle);
        if (status != 0 || handle.IsInvalid)
        {
            string message = NativeMethods.GetLastError();
            handle?.Dispose();
            throw new Yolo26Exception(status == 0 ? -1 : status, message);
        }

        detector = handle;
    }

    public ImagePrediction Predict(BgrImage image)
    {
        ArgumentNullException.ThrowIfNull(image);
        BatchPrediction batch = Predict(new[] { image });
        return new ImagePrediction(batch.Detections[0], batch.Timing);
    }

    public BatchPrediction Predict(IReadOnlyList<BgrImage> images)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
        {
            throw new ArgumentException("至少需要一张图片。", nameof(images));
        }

        lock (syncRoot)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return PredictCore(images);
        }
    }

    public void Dispose()
    {
        lock (syncRoot)
        {
            if (disposed)
            {
                return;
            }

            detector.Dispose();
            disposed = true;
        }
    }

    private BatchPrediction PredictCore(IReadOnlyList<BgrImage> images)
    {
        NativeImage[] nativeImages = new NativeImage[images.Count];
        GCHandle[] pins = new GCHandle[images.Count];
        try
        {
            for (int i = 0; i < images.Count; ++i)
            {
                BgrImage image = images[i]
                    ?? throw new ArgumentException($"第 {i} 张图片为空。", nameof(images));
                pins[i] = GCHandle.Alloc(image.Data, GCHandleType.Pinned);
                nativeImages[i] = new NativeImage
                {
                    Data = pins[i].AddrOfPinnedObject(),
                    Width = image.Width,
                    Height = image.Height,
                    Stride = checked((ulong)image.Stride)
                };
            }

            int status = NativeMethods.Predict(detector, nativeImages, nativeImages.Length,
                out ResultSafeHandle result);
            using (result)
            {
                CheckStatus(status);
                if (result.IsInvalid)
                {
                    ThrowNativeError(-1);
                }

                Detection[][] detections = ReadDetections(result, images.Count);
                CheckStatus(NativeMethods.GetTiming(result, out NativeTiming nativeTiming));
                InferenceTiming timing = new(
                    nativeTiming.PreprocessMs,
                    nativeTiming.InferenceMs,
                    nativeTiming.PostprocessMs,
                    nativeTiming.TotalMs);
                return new BatchPrediction(detections, timing);
            }
        }
        finally
        {
            foreach (GCHandle pin in pins)
            {
                if (pin.IsAllocated)
                {
                    pin.Free();
                }
            }
        }
    }

    private static Detection[][] ReadDetections(ResultSafeHandle result, int imageCount)
    {
        Detection[][] batch = new Detection[imageCount][];
        int nativeSize = Marshal.SizeOf<NativeDetection>();

        for (int imageIndex = 0; imageIndex < imageCount; ++imageIndex)
        {
            CheckStatus(NativeMethods.GetResult(
                result, imageIndex, out IntPtr nativeDetections, out int count));
            if (count < 0 || (count > 0 && nativeDetections == IntPtr.Zero))
            {
                throw new InvalidOperationException("原生检测结果的内存布局无效。");
            }

            Detection[] detections = new Detection[count];
            for (int i = 0; i < count; ++i)
            {
                int offset = checked(i * nativeSize);
                NativeDetection detection = Marshal.PtrToStructure<NativeDetection>(
                    IntPtr.Add(nativeDetections, offset));
                detections[i] = new Detection(
                    detection.Left,
                    detection.Top,
                    detection.Right,
                    detection.Bottom,
                    detection.Confidence,
                    detection.ClassId);
            }

            batch[imageIndex] = detections;
        }

        return batch;
    }

    private static void CheckStatus(int status)
    {
        if (status != 0)
        {
            ThrowNativeError(status);
        }
    }

    private static void ThrowNativeError(int status)
    {
        throw new Yolo26Exception(status, NativeMethods.GetLastError());
    }
}
