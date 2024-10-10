from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(
        r"D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\model\pt\best.pt")  # load a custom model

    # Predict with the model
    results = model(r"D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\images\d_451_2024-09-26-09-34-04_c04_train.jpg", save=True, imgsz=1024, conf=0.2, iou=0.4, agnostic_nms=True)  # predict on an image
    print(results)