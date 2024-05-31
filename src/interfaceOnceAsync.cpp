#include <opencv2/opencv.hpp>
#include "infer.hpp"
#include "cpm.hpp"
#include "yolo.hpp"

using namespace std;

const int width = 1920;
const int height = 1200;

cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;

bool initAsync(const string &engine_file) {
    bool ok = cpmi.start([&engine_file] {
        return yolo::load(engine_file, yolo::Type::V8);
    }, 1);
    if (!ok) {
        cout << "================================= TensorRT INIT FAIL =================================" << endl;
        return false;
    } else {
        cout << "================================= TensorRT INIT SUCCESS =================================" << endl;
        return true;
    }
}

vector<yolo::Box> inferAsync(uchar *image) {
    trt::Timer timer;
    auto img = yolo::Image(image, width, height);
    timer.start();
    auto objs = cpmi.commit(img).get();
    timer.stop("BATCH1");
    return objs;
}

extern "C" __declspec(dllexport) bool TensorRT_INIT_ASYNC(const char *engine_file) {
    return initAsync(engine_file);
}

// // ����list<box>�Ĵ�С
// // �ɹ���������  ʧ������
// int BeiginGetListBoxData();
//
// // ����listbox������  ��Ҫ��ε���
// // boxLable list<box>������ ��0��size
// int GetListBoxData(int boxLable, float *left, float *top, float *right, float *bottom, float *confidence, int *classLabel);
//
// // ��������list<box>����  ���c++�ڲ���Ҫ�ͷ���Դ�����������ͷž�����
// void EndGetListBoxData();

static std::vector<yolo::Box> g_boxes;

extern "C" __declspec(dllexport) int TensorRT_INFER_ASYNC(uchar *image) {
    g_boxes = inferAsync(image);
    int size = g_boxes.size();
    // yolo::Box *result = new yolo::Box[boxes.size()];
    // std::copy(boxes.begin(), boxes.end(), result);
    //
    // if (size <= 0) {
    //     return -1;
    // }
    // g_boxes.assign(result, result + size);
    // delete[] result;
    return size;
}

extern "C" __declspec(dllexport) int GetListBoxData(int boxLabel, float *left, float *top, float *right, float *bottom,
                                                    float *confidence, int *classLabel) {
    if (boxLabel < 0 || boxLabel >= g_boxes.size()) {
        return -1;
    }
    const yolo::Box &box = g_boxes[boxLabel];
    *left = box.left;
    *top = box.top;
    *right = box.right;
    *bottom = box.bottom;
    *confidence = box.confidence;
    *classLabel = box.class_label;
    return 0;
}

extern "C" __declspec(dllexport) void EndGetListBoxData() {
    g_boxes.clear();
}
