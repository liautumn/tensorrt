#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

#include "string"

using namespace std;

class Config {
public:
    const string MODEL = R"(D:\autumn\Documents\WeChat Files\wxid_1w2acnt5bx6s22\FileStorage\File\2025-04\best7.2_withNMS.engine)";
    const string TEST_IMG = R"(D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\images\207_1734507046.jpeg)";
};

#endif //YOLO_CONFIG_H
