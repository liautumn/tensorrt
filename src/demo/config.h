#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

#include "string"

using namespace std;

class Config {
public:
    const string MODEL = R"(D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\model\engine\best.transd.engine)";
    const string TEST_IMG = R"(D:\autumn\Documents\WeChat Files\wxid_1w2acnt5bx6s22\FileStorage\File\2024-10\10\d_154_2024-10-14-09-30-00_c10.jpg)";
    const string TEST_IMG_DIRECTORY = R"(D:\autumn\Documents\WeChat Files\wxid_1w2acnt5bx6s22\FileStorage\File\2024-10\10)";

    const char *labels[22] = {
            "BB", "ZH", "ZK", "JK", "ZZ", "GS", "ZW", "DJ", "PD", "CS", "DW", "HN",
            "YW", "FH", "LZ", "SYQ", "BQ", "DPD", "MD", "CH", "SD", "SZ"
    };

};

#endif //YOLO_CONFIG_H
