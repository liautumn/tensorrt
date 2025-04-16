#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

using namespace std;

class Config {
public:
    const string MODEL = "/home/autumn/Documents/GitHub/tensorrt/workspace/model/engine/yolo11s.dynamic.transd.engine";
    const string TEST_IMG = "/home/autumn/Documents/GitHub/tensorrt/workspace/images/zidane.jpg";
    const string VIDEO_PATH = "/home/autumn/Documents/GitHub/tensorrt/workspace/images/001.mp4";
};

#endif //YOLO_CONFIG_H
