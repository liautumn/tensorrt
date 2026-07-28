using System;
using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace Yolo26Net;

[StructLayout(LayoutKind.Sequential, Pack = 8, Size = 24)]
internal struct NativeImage {
  internal IntPtr Data;
  internal int Width;
  internal int Height;
  internal ulong Stride;
}

[StructLayout(LayoutKind.Sequential, Pack = 8, Size = 24)]
internal struct NativeDetection {
  internal float Left;
  internal float Top;
  internal float Right;
  internal float Bottom;
  internal float Confidence;
  internal int ClassId;
}

[StructLayout(LayoutKind.Sequential, Pack = 4, Size = 16)]
internal struct NativeTiming {
  internal float PreprocessMs;
  internal float InferenceMs;
  internal float PostprocessMs;
  internal float TotalMs;
}

internal sealed class DetectorSafeHandle : SafeHandleZeroOrMinusOneIsInvalid {
  public DetectorSafeHandle() : base(true) {}

  protected override bool ReleaseHandle() {
    NativeMethods.DestroyDetector(handle);
    return true;
  }
}

internal sealed class ResultSafeHandle : SafeHandleZeroOrMinusOneIsInvalid {
  public ResultSafeHandle() : base(true) {}

  protected override bool ReleaseHandle() {
    NativeMethods.DestroyResult(handle);
    return true;
  }
}

internal sealed class CpmSafeHandle : SafeHandleZeroOrMinusOneIsInvalid {
  public CpmSafeHandle() : base(true) {}

  protected override bool ReleaseHandle() {
    NativeMethods.DestroyCpm(handle);
    return true;
  }
}

internal static class NativeMethods {
  private const string LibraryName = "yolo26";

  static NativeMethods() {
    if (IntPtr.Size != 8 || Marshal.SizeOf<NativeImage>() != 24 ||
        Marshal.SizeOf<NativeDetection>() != 24 ||
        Marshal.SizeOf<NativeTiming>() != 16) {
      throw new PlatformNotSupportedException(
          "YOLO26 C# 调用层仅支持 x64 ABI。");
    }

    if (!string.IsNullOrWhiteSpace(Config.NativeLibraryPath)) {
      NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly,
                                         ResolveNativeLibrary);
    }
  }

  [DllImport(LibraryName, EntryPoint = "yolo26_detector_create",
             ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int
  CreateDetector([MarshalAs(UnmanagedType.LPUTF8Str)] string enginePath,
                 float confidenceThreshold, int gpuDevice,
                 out DetectorSafeHandle detector);

  [DllImport(LibraryName, EntryPoint = "yolo26_detector_destroy",
             ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern void DestroyDetector(IntPtr detector);

  [DllImport(LibraryName, EntryPoint = "yolo26_detector_predict",
             ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int Predict(DetectorSafeHandle detector,
                                     [In] NativeImage[] images, int imageCount,
                                     out ResultSafeHandle result);

  [DllImport(LibraryName, EntryPoint = "yolo26_result_get",
             ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int GetResult(ResultSafeHandle result, int imageIndex,
                                       out IntPtr detections, out int count);

  [DllImport(LibraryName, EntryPoint = "yolo26_result_get_timing",
             ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int GetTiming(ResultSafeHandle result,
                                       out NativeTiming timing);

  [DllImport(LibraryName, EntryPoint = "yolo26_result_destroy",
             ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern void DestroyResult(IntPtr result);

  [DllImport(LibraryName, EntryPoint = "yolo26_cpm_create",
             ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int
  CreateCpm([MarshalAs(UnmanagedType.LPUTF8Str)] string enginePath,
            float confidenceThreshold, int gpuDevice, int maxBatchSize,
            out CpmSafeHandle cpm);

  [DllImport(LibraryName, EntryPoint = "yolo26_cpm_predict_one",
             ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int PredictCpm(CpmSafeHandle cpm, in NativeImage image,
                                        out IntPtr detections, out int count);

  [DllImport(LibraryName, EntryPoint = "yolo26_cpm_destroy",
             ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern void DestroyCpm(IntPtr cpm);

  [DllImport(LibraryName, EntryPoint = "yolo26_last_error",
             ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  private static extern IntPtr LastError();

  internal static string GetLastError() {
    IntPtr message = LastError();
    return message == IntPtr.Zero
               ? "没有原生错误信息。"
               : Marshal.PtrToStringUTF8(message) ?? "无法读取原生错误信息。";
  }

  private static IntPtr ResolveNativeLibrary(string libraryName,
                                             Assembly assembly,
                                             DllImportSearchPath? searchPath) {
    return libraryName == LibraryName
               ? NativeLibrary.Load(Config.NativeLibraryPath)
               : IntPtr.Zero;
  }
}
