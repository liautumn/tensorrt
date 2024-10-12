from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("../model/pt/best.pt")

    # Customize validation settings
    validation_results = model.val(data="D:/autumn/Desktop/zhiwu.yaml", imgsz=1024, batch=4, conf=0.2, iou=0.4, device="0")