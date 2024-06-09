from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # load an official model

# Export the model
# ONNX ===> imgsz, half, dynamic, simplify, opset, batch
# TensorRT ===> imgsz, half, dynamic, simplify, workspace, int8, batch
model.export(format='onnx',
             imgsz=(1024, 1024),
             # half=True,
             dynamic=True,
             simplify=True,
             # batch=1
             )

# model.export(format='engine',
#              imgsz=(1024, 1024),
#              half=False,
#              dynamic=False,
#              simplify=True,
#              workspace=4.0,
#              int8=True,
#              batch=1)
