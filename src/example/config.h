#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

#include "string"

using namespace std;

class Config {
public:
    const string MODEL = R"(D:\autumn\Documents\GitHub\tensorrt\workspace\model\engine\yolo11s.dynamic.transd.engine)";
    const string TEST_IMG = R"(D:\autumn\Documents\GitHub\tensorrt\workspace\images\207_1734507046.jpeg)";
};

#endif //YOLO_CONFIG_H
