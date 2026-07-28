#ifndef YOLO26_C_H
#define YOLO26_C_H

#include <stdint.h>

#if defined(_WIN32)
#if defined(YOLO26_BUILD_DLL)
#define YOLO26_API __declspec(dllexport)
#else
#define YOLO26_API __declspec(dllimport)
#endif
#else
#define YOLO26_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yolo26_detector yolo26_detector;
typedef struct yolo26_result yolo26_result;
typedef struct yolo26_cpm yolo26_cpm;

typedef enum yolo26_status {
  YOLO26_OK = 0,
  YOLO26_INVALID_ARGUMENT = 1,
  YOLO26_OUT_OF_MEMORY = 2,
  YOLO26_RUNTIME_ERROR = 3,
  YOLO26_UNKNOWN_ERROR = 4
} yolo26_status;

typedef struct yolo26_image {
  const uint8_t *bgr;
  int32_t width;
  int32_t height;
  uint64_t stride;
} yolo26_image;

typedef struct yolo26_detection {
  float left;
  float top;
  float right;
  float bottom;
  float confidence;
  int32_t class_id;
} yolo26_detection;

typedef struct yolo26_timing {
  float preprocess_ms;
  float inference_ms;
  float postprocess_ms;
  float total_ms;
} yolo26_timing;

YOLO26_API int32_t yolo26_detector_create(const char *engine_file_utf8,
                                          float confidence, int32_t gpu_device,
                                          yolo26_detector **out_detector);
YOLO26_API void yolo26_detector_destroy(yolo26_detector *detector);

// 同一个 detector 不允许并发 predict。images 与 BGR 缓冲在调用返回前必须保持有效且
// 不可修改。销毁句柄前，调用方必须等待相关调用全部结束。
YOLO26_API int32_t yolo26_detector_predict(yolo26_detector *detector,
                                           const yolo26_image *images,
                                           int32_t image_count,
                                           yolo26_result **out_result);
YOLO26_API int32_t yolo26_result_get(const yolo26_result *result,
                                     int32_t image_index,
                                     const yolo26_detection **out_detections,
                                     int32_t *out_count);
YOLO26_API int32_t yolo26_result_get_timing(const yolo26_result *result,
                                            yolo26_timing *out_timing);
YOLO26_API void yolo26_result_destroy(yolo26_result *result);

// CPM 接口适用于多个线程持续提交单张图片，内部自动组成 N 张 batch。
// engine 的最小 batch 必须为 1，max_batch_size 不能超过 profile 上限。
// image 与 BGR 缓冲在 predict_one 返回前必须保持有效且不可修改。
// 返回的 detections 是线程局部借用内存，不需要释放；同一线程再次调用任意 CPM
// predict_one 或线程退出后即失效，不能交给其他线程持有。
// 销毁 CPM 前，调用方必须停止提交并等待所有 predict_one 调用结束。
YOLO26_API int32_t yolo26_cpm_create(const char *engine_file_utf8,
                                     float confidence, int32_t gpu_device,
                                     int32_t max_batch_size,
                                     yolo26_cpm **out_cpm);
YOLO26_API int32_t yolo26_cpm_predict_one(
    yolo26_cpm *cpm, const yolo26_image *image,
    const yolo26_detection **out_detections, int32_t *out_count);
YOLO26_API void yolo26_cpm_destroy(yolo26_cpm *cpm);

YOLO26_API const char *yolo26_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // YOLO26_C_H
