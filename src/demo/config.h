#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

#include "string"

using namespace std;

class Config {
public:
    const string MODEL = R"(D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\model\engine\best.transd.engine)";
    const string TEST_IMG = R"(D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\images\d_451_2024-09-26-09-34-04_c04_train.jpg)";

    const char *labels[22] = {
            "BB", "ZH", "ZK", "JK", "ZZ", "GS", "ZW", "DJ", "PD", "CS", "DW", "HN",
            "YW", "FH", "LZ", "SYQ", "BQ", "DPD", "MD", "CH", "SD", "SZ"
    };

};

#endif //YOLO_CONFIG_H
