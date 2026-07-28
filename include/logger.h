#ifndef YOLO26_LOGGER_H
#define YOLO26_LOGGER_H

#include <string>

namespace yolo26::detail {

enum class LogLevel { info, warning, error };

void log(LogLevel level, const std::string &message) noexcept;

} // namespace yolo26::detail

#endif // YOLO26_LOGGER_H
