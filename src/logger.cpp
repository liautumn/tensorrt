#include "logger.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace yolo26::detail {
namespace {

std::mutex log_mutex;

std::tm local_time(std::time_t value) {
  std::tm result{};
#ifdef _WIN32
  localtime_s(&result, &value);
#else
  localtime_r(&value, &result);
#endif
  return result;
}

const char *level_name(LogLevel level) {
  switch (level) {
  case LogLevel::info:
    return "INFO";
  case LogLevel::warning:
    return "WARN";
  case LogLevel::error:
    return "ERROR";
  }
  return "INFO";
}

} // namespace

void log(LogLevel level, const std::string &message) noexcept {
  try {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    const std::tm local = local_time(time);

    std::ostringstream file_name;
    file_name << std::put_time(&local, "%Y-%m-%d") << "-trt.log";

    std::ostringstream line;
    line << std::put_time(&local, "%Y-%m-%d %H:%M:%S") << " ["
         << level_name(level) << "] " << message;

    std::lock_guard<std::mutex> lock(log_mutex);
    std::cout << line.str() << '\n';
    std::ofstream file(file_name.str(), std::ios::app);
    if (file) {
      file << line.str() << '\n';
    }
  } catch (...) {
    // 日志失败不能中断推理。
  }
}

} // namespace yolo26::detail
