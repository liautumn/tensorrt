using System;

namespace Yolo26Net;

public readonly struct Detection
{
    internal Detection(float left, float top, float right, float bottom, float confidence, int classId)
    {
        Left = left;
        Top = top;
        Right = right;
        Bottom = bottom;
        Confidence = confidence;
        ClassId = classId;
    }

    public float Left { get; }
    public float Top { get; }
    public float Right { get; }
    public float Bottom { get; }
    public float Confidence { get; }
    public int ClassId { get; }
}

public readonly struct InferenceTiming
{
    internal InferenceTiming(float preprocessMs, float inferenceMs, float postprocessMs, float totalMs)
    {
        PreprocessMs = preprocessMs;
        InferenceMs = inferenceMs;
        PostprocessMs = postprocessMs;
        TotalMs = totalMs;
    }

    public float PreprocessMs { get; }
    public float InferenceMs { get; }
    public float PostprocessMs { get; }
    public float TotalMs { get; }
}

public sealed class ImagePrediction
{
    internal ImagePrediction(Detection[] detections, InferenceTiming timing)
    {
        Detections = detections;
        Timing = timing;
    }

    public Detection[] Detections { get; }
    public InferenceTiming Timing { get; }
}

public sealed class BatchPrediction
{
    internal BatchPrediction(Detection[][] detections, InferenceTiming timing)
    {
        Detections = detections;
        Timing = timing;
    }

    public Detection[][] Detections { get; }
    public InferenceTiming Timing { get; }
}

public sealed class Yolo26Exception : Exception
{
    internal Yolo26Exception(int status, string message)
        : base($"YOLO26 原生调用失败（状态码 {status}）：{message}")
    {
        Status = status;
    }

    public int Status { get; }
}
