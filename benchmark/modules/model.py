import numpy as np
from ultralytics import YOLO


class YOLOv8:

    def __init__(self, model_cfg):
        self.model = YOLO(model_cfg["model_path"])
        self.names = self.model.names
        self.conf_threshold = model_cfg["conf_threshold"]
        self.iou_threshold = model_cfg["iou_threshold"]


    def raw_infer(self, image):
        return self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)


    def infer(self, image):
        """
            return:
                predictions = list({
                    "xyxy": np.int16 with shape (4,)
                    "cls_id": np.int16 number
                    "conf": float
                })
        """
        boxes = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)[0].boxes
        cls_list = boxes.cls.cpu().numpy().astype(np.int16)
        xyxy_list = boxes.xyxy.cpu().numpy().astype(np.int16)
        conf_list = boxes.conf.cpu().numpy()

        predictions = list()
        zip_values = zip(cls_list, xyxy_list, conf_list)
        for (cls_value, xyxy_value, conf_value) in zip_values:
            predictions.append({
                "xyxy": xyxy_value,
                "cls_id": cls_value,
                "cls_name": self.names[cls_value],
                "conf": conf_value,
            })

        return predictions
    