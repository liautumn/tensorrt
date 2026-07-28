using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace Yolo26Net;

// 多个线程可同时调用 Predict，native CPM 会把请求自动组成 N 张 batch。
public sealed class Yolo26Cpm : IDisposable {
  private readonly CpmSafeHandle cpm;
  private int disposed;

  public Yolo26Cpm(string enginePath, int maxBatchSize,
                   float confidenceThreshold = 0.25F, int gpuDevice = 0) {
    if (string.IsNullOrWhiteSpace(enginePath)) {
      throw new ArgumentException("Engine 路径不能为空。", nameof(enginePath));
    }

    if (maxBatchSize <= 0) {
      throw new ArgumentOutOfRangeException(nameof(maxBatchSize));
    }

    int status =
        NativeMethods.CreateCpm(enginePath, confidenceThreshold, gpuDevice,
                                maxBatchSize, out CpmSafeHandle handle);
    if (status != 0 || handle.IsInvalid) {
      string message = NativeMethods.GetLastError();
      handle?.Dispose();
      throw new Yolo26Exception(status == 0 ? -1 : status, message);
    }

    cpm = handle;
  }

  public Detection[] Predict(BgrImage image) {
    ArgumentNullException.ThrowIfNull(image);
    ObjectDisposedException.ThrowIf(Volatile.Read(ref disposed) != 0, this);

    GCHandle pin = GCHandle.Alloc(image.Data, GCHandleType.Pinned);
    try {
      NativeImage native = new() { Data = pin.AddrOfPinnedObject(),
                                   Width = image.Width, Height = image.Height,
                                   Stride = checked((ulong)image.Stride) };
      int status = NativeMethods.PredictCpm(cpm, in native, out IntPtr pointer,
                                            out int count);
      if (status != 0) {
        throw new Yolo26Exception(status, NativeMethods.GetLastError());
      }

      if (count < 0 || (count > 0 && pointer == IntPtr.Zero)) {
        throw new InvalidOperationException("原生 CPM 结果的内存布局无效。");
      }

      Detection[] result = new Detection[count];
      int size = Marshal.SizeOf<NativeDetection>();
      for (int index = 0; index < count; ++index) {
        NativeDetection item = Marshal.PtrToStructure<NativeDetection>(
            IntPtr.Add(pointer, checked(index * size)));
        result[index] =
            new Detection(item.Left, item.Top, item.Right, item.Bottom,
                          item.Confidence, item.ClassId);
      }
      return result;
    } finally {
      pin.Free();
    }
  }

  public void Dispose() {
    if (Interlocked.Exchange(ref disposed, 1) != 0) {
      return;
    }

    cpm.Dispose();
  }
}
