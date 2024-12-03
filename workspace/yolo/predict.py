from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(
        r"D:\autumn\Downloads\best.pt")  # load a custom model

    # Predict with the model
    results = model(r"F:\FlawImages2\b8d95714-2b8c-4959-92fa-959ce86cc50f.jpeg", save=True, imgsz=1024, conf=0.2, iou=0.2, agnostic_nms=True)  # predict on an image
    print(results)