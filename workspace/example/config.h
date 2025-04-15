#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

using namespace std;

class Config {
public:
    const string MODEL = "D:/autumn/Documents/GitHub/tensorrt/workspace/model/engine/yolo11s.transd.engine";
    // const string MODEL = "D:/autumn/Documents/GitHub/tensorrt/workspace/model/engine/yolo11s-obb.transd.engine";
    const string TEST_IMG = "D:/autumn/Documents/GitHub/tensorrt/workspace/images/P0009.jpg";
    const string VIDEO_PATH = "D:/autumn/Downloads/001.mp4";
};

#endif //YOLO_CONFIG_H
