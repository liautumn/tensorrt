using System;
using System.IO;

namespace Yolo26Net;

public static class Config
{
    // 根 CMake 会把动态库输出到 workspace/export，C# 直接按绝对路径加载。
    public static readonly string NativeLibraryPath = Path.Combine(
        ProjectRoot,
        "workspace",
        "export",
        OperatingSystem.IsWindows() ? "yolo26.dll" : "libyolo26.so");

    // 所有示例参数都在这里设置，不需要命令行参数。
    public static readonly string EnginePath =
        Path.Combine(ProjectRoot, "workspace", "model", "engine", "yolo26.engine");

    // 一张图片是单图推理，多张图片会按实际数量 N 批量推理。
    public static readonly string[] ImagePaths =
    {
        Path.Combine(ProjectRoot, "workspace", "images", "bus.jpg")
    };

    public static readonly string OutputDirectory =
        Path.Combine(ProjectRoot, "workspace", "output");

    public const float ConfidenceThreshold = 0.25F;
    public const int GpuDevice = 0;

    // 自定义模型请按训练数据集的类别顺序修改。
    public static readonly string[] ClassNames =
    {
        "BB", "ZH", "ZK", "JK", "ZZ", "GS", "ZW", "DJ", "PD", "CS", "DW", "HN",
        "YW", "FH", "LZ", "SYQ", "BQ", "DPD", "MD", "CH", "SD", "SZ", "ZS"
    };

    public static string GetClassName(int classId)
    {
        return classId >= 0 && classId < ClassNames.Length
            ? ClassNames[classId]
            : classId.ToString();
    }

    private static string ProjectRoot => FindProjectRoot();

    private static string FindProjectRoot()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory != null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, "workspace")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return Directory.GetCurrentDirectory();
    }
}
