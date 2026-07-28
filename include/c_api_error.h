#ifndef YOLO26_C_API_ERROR_H
#define YOLO26_C_API_ERROR_H

#include "yolo26_c.h"

#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <utility>

namespace yolo26::detail {

void clear_c_api_error() noexcept;
void set_c_api_error(const char *message) noexcept;

template <typename Function> int32_t c_api_call(Function &&function) noexcept {
  clear_c_api_error();
  try {
    std::forward<Function>(function)();
    return YOLO26_OK;
  } catch (const std::invalid_argument &error) {
    set_c_api_error(error.what());
    return YOLO26_INVALID_ARGUMENT;
  } catch (const std::bad_alloc &error) {
    set_c_api_error(error.what());
    return YOLO26_OUT_OF_MEMORY;
  } catch (const std::exception &error) {
    set_c_api_error(error.what());
    return YOLO26_RUNTIME_ERROR;
  } catch (...) {
    set_c_api_error("unknown native exception");
    return YOLO26_UNKNOWN_ERROR;
  }
}

} // namespace yolo26::detail

#endif // YOLO26_C_API_ERROR_H
