#ifndef YOLO26_CHECKED_MATH_H
#define YOLO26_CHECKED_MATH_H

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace yolo26::detail {

inline std::size_t checked_multiply(std::size_t left, std::size_t right,
                                    const char *message) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error(message);
  }
  return left * right;
}

} // namespace yolo26::detail

#endif // YOLO26_CHECKED_MATH_H
