#include "tensorrt_engine.h"

#include "logger.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace yolo26::detail {
namespace {

std::vector<char> read_engine_plan(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open engine file: " + path);
  }

  const std::streamsize length = input.tellg();
  if (length <= 0) {
    throw std::runtime_error("engine file is empty: " + path);
  }
  input.seekg(0, std::ios::beg);

  std::vector<char> file(static_cast<std::size_t>(length));
  if (!input.read(file.data(), length)) {
    throw std::runtime_error("cannot read engine file: " + path);
  }

  // Ultralytics engine files may start with: uint32 JSON length + JSON metadata
  // + TensorRT plan.
  std::size_t plan_offset = 0;
  if (file.size() > sizeof(std::uint32_t)) {
    std::uint32_t metadata_length = 0;
    std::memcpy(&metadata_length, file.data(), sizeof(metadata_length));
    const std::size_t candidate = sizeof(metadata_length) + metadata_length;
    if (metadata_length >= 2 && candidate < file.size() &&
        file[sizeof(metadata_length)] == '{' && file[candidate - 1] == '}') {
      plan_offset = candidate;
    }
  }

  return {file.begin() + static_cast<std::ptrdiff_t>(plan_offset), file.end()};
}

std::vector<int> to_vector(const nvinfer1::Dims &dims) {
  if (dims.nbDims < 0) {
    throw std::runtime_error("TensorRT returned an invalid tensor shape");
  }
  return {dims.d, dims.d + dims.nbDims};
}

nvinfer1::Dims to_dims(const std::vector<int> &shape) {
  if (shape.size() > 8) {
    throw std::invalid_argument("TensorRT shape has too many dimensions");
  }
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int>(shape.size());
  for (int i = 0; i < dims.nbDims; ++i) {
    dims.d[i] = shape[i];
  }
  return dims;
}

} // namespace

void TensorRTLogger::log(Severity severity, const char *message) noexcept {
  try {
    if (severity <= Severity::kERROR) {
      detail::log(LogLevel::error, std::string("TensorRT: ") + message);
    } else if (severity <= Severity::kWARNING) {
      detail::log(LogLevel::warning, std::string("TensorRT: ") + message);
    }
  } catch (...) {
  }
}

TensorRTEngine::TensorRTEngine(const std::string &engine_file) {
  const std::vector<char> plan = read_engine_plan(engine_file);
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    throw std::runtime_error("cannot create TensorRT runtime");
  }

  engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine_) {
    throw std::runtime_error("cannot deserialize TensorRT engine; check the "
                             "TensorRT version and GPU target");
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("cannot create TensorRT execution context");
  }

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const std::string name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name.c_str()) ==
        nvinfer1::TensorIOMode::kINPUT) {
      if (!input_name_.empty()) {
        throw std::runtime_error(
            "YOLO26 detection engine must have exactly one input tensor");
      }
      input_name_ = name;
    } else {
      if (!output_name_.empty()) {
        throw std::runtime_error(
            "YOLO26 detection engine must have exactly one output tensor");
      }
      output_name_ = name;
    }
  }
  if (input_name_.empty() || output_name_.empty()) {
    throw std::runtime_error(
        "YOLO26 detection engine must have one input and one output tensor");
  }
}

std::vector<int> TensorRTEngine::declared_shape(const std::string &name) const {
  return to_vector(engine_->getTensorShape(name.c_str()));
}

std::vector<int> TensorRTEngine::runtime_shape(const std::string &name) const {
  return to_vector(context_->getTensorShape(name.c_str()));
}

std::vector<int>
TensorRTEngine::profile_shape(const std::string &name,
                              nvinfer1::OptProfileSelector selector) const {
  return to_vector(engine_->getProfileShape(name.c_str(), 0, selector));
}

nvinfer1::DataType TensorRTEngine::data_type(const std::string &name) const {
  return engine_->getTensorDataType(name.c_str());
}

nvinfer1::TensorFormat
TensorRTEngine::tensor_format(const std::string &name) const {
  return engine_->getTensorFormat(name.c_str());
}

void TensorRTEngine::set_input_shape(const std::vector<int> &shape) {
  if (!context_->setInputShape(input_name_.c_str(), to_dims(shape))) {
    throw std::runtime_error("TensorRT rejected input shape " +
                             format_shape(shape));
  }
}

void TensorRTEngine::set_tensor_addresses(void *input, void *output) {
  if (!context_->setTensorAddress(input_name_.c_str(), input) ||
      !context_->setTensorAddress(output_name_.c_str(), output)) {
    throw std::runtime_error(
        "TensorRT rejected an input or output buffer address");
  }
}

void TensorRTEngine::enqueue(cudaStream_t stream) {
  if (!context_->enqueueV3(stream)) {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }
}

std::string format_shape(const std::vector<int> &shape) {
  std::ostringstream output;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      output << 'x';
    }
    output << shape[i];
  }
  return output.str();
}

} // namespace yolo26::detail
