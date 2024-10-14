# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm
import os.path as osp

# 定义标签的类别ID映射
label_to_id = {
    "BB": 0,
    "ZH": 1,
    "ZK": 2,
    "JK": 3,
    "ZZ": 4,
    "GS": 5,
    "ZW": 6,
    "DJ": 7,
    "PD": 8,
    "CS": 9,
    "DW": 10,
    "HN": 11,
    "YW": 12,
    "FH": 13,
    "LZ": 14,
    "SYQ": 15,
    "BQ": 16,
    "DPD": 17,
    "MD": 18,
    "CH": 19,
    "SD": 20,
    "SZ": 21
}


def custom_to_yolov5(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    with open(output_file, "w", encoding="utf-8") as f:
        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]

            class_index = label_to_id.get(label)

            x_center = (points[0][0] + points[2][0]) / (2 * image_width)
            y_center = (points[0][1] + points[2][1]) / (2 * image_height)
            width = abs(points[2][0] - points[0][0]) / image_width
            height = abs(points[2][1] - points[0][1]) / image_height

            f.write(
                f"{class_index} {x_center} {y_center} {width} {height}\n"
            )


if __name__ == '__main__':
    # 调用函数，传入文件夹路径
    input_path = r'D:\autumn\Pictures\test\1'
    output_path = r'D:\autumn\Pictures\test\1\out'
    file_list = os.listdir(input_path)
    os.makedirs(output_path, exist_ok=True)
    for file_name in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
    ):
        if not file_name.endswith(".json"):
            continue
        src_file = osp.join(input_path, file_name)
        dst_file = osp.join(
            output_path, osp.splitext(file_name)[0] + ".txt"
        )
        custom_to_yolov5(src_file, dst_file)
