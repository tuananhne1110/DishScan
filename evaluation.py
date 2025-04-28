from ultralytics import YOLO

model = YOLO(r'./checkpoints/dish_scan_v6.pt') 

metrics = model.val(data=r'./datasets/data.yaml', conf=0.8, iou=0.8, save_json=True)

metrics.confusion_matrix.plot(
    normalize=True,
    names=model.names,
    save_dir='.',
)