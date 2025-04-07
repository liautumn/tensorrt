#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

#include "string"

using namespace std;

class Config {
public:
    const string MODEL = R"(D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\model\engine\best8.0.engine)";
    const string TEST_IMG = R"(D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\images\207_1734507046.jpeg)";
};

#endif //YOLO_CONFIG_H
