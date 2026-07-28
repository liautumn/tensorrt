#ifndef YOLO26_TENSORRT_ENGINE_H
#define YOLO26_TENSORRT_ENGINE_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

namespace yolo26::detail {

class TensorRTLogger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *message) noexcept override;
};

class TensorRTEngine {
public:
  explicit TensorRTEngine(const std::string &engine_file);

  TensorRTEngine(const TensorRTEngine &) = delete;
  TensorRTEngine &operator=(const TensorRTEngine &) = delete;

  const std::string &input_name() const noexcept { return input_name_; }
  const std::string &output_name() const noexcept { return output_name_; }

  std::vector<int> declared_shape(const std::string &name) const;
  std::vector<int> runtime_shape(const std::string &name) const;
  std::vector<int> profile_shape(const std::string &name,
                                 nvinfer1::OptProfileSelector selector) const;
  nvinfer1::DataType data_type(const std::string &name) const;
  nvinfer1::TensorFormat tensor_format(const std::string &name) const;

  void set_input_shape(const std::vector<int> &shape);
  void set_tensor_addresses(void *input, void *output);
  void enqueue(cudaStream_t stream);

private:
  TensorRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::string input_name_;
  std::string output_name_;
};

std::string format_shape(const std::vector<int> &shape);

} // namespace yolo26::detail

#endif // YOLO26_TENSORRT_ENGINE_H
