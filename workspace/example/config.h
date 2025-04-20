#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

using namespace std;

class Config {
public:
    const int GPU_DEVICE = 0;
    const string MODEL = R"(D:\autumn\Documents\GitHub\tensorrt\workspace\model\engine\yolo11s-cls.engine)";
    const string TEST_IMG = R"(D:\autumn\Documents\GitHub\tensorrt\workspace\images\bus.jpg)";
    const string VIDEO_PATH = R"(D:\autumn\Downloads\1144260702_nb3-1-16.mp4)";
};

#endif //YOLO_CONFIG_H
