#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

using namespace std;

class Config {
public:
    const int GPU_DEVICE = 0;
    const string MODEL = R"(/home/autumn/Documents/GitHub/tensorrt/workspace/model/engine/yolo11s-pose.transd.engine)";
    const string TEST_IMG = R"(/home/autumn/Documents/GitHub/tensorrt/workspace/images/bus.jpg)";
    const string VIDEO_PATH = R"(/home/autumn/Documents/GitHub/tensorrt/workspace/images/002.mp4)";
};

#endif //YOLO_CONFIG_H
