#ifndef YOLO26_EXAMPLE_CONFIG_H
#define YOLO26_EXAMPLE_CONFIG_H

#include <string>
#include <vector>

namespace example {

struct Config {
  // 修改这里即可运行示例，不需要命令行参数。
  std::string engine_file = "workspace/model/engine/yolo26n.engine";
  std::vector<std::string> image_files = {
      "workspace/images/bus.jpg",
  };
  std::string output_directory = "workspace/output";
  float confidence_threshold = 0.25F;
  int gpu_device = 0;
  bool save_images = true;
  bool print_detections = true;
};

} // namespace example

#endif // YOLO26_EXAMPLE_CONFIG_H
