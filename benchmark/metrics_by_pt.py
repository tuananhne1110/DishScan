import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from kano.dataset_utils import YoloImage

from modules.utils import read_yaml_file
from modules.yolov8 import YOLOv8


def get_gt_boxes(image_path, gt_remap_dict):
    yolo_img = YoloImage(image_path)
    gt_boxes = [
        {
            "bbox": np.array(l["xyxy"]),
            "label": gt_remap_dict[l["class"]],

        }
        for l in yolo_img.get_labels()
    ]
    return gt_boxes  


def create_coco_format(gt_boxes, det_boxes):
    coco_gt_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": str(i)} for i in range(4)],
    }
    coco_det_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": str(i)} for i in range(4)],
    }

    annotation_id = 0

    for i, (gt, det) in enumerate(zip(gt_boxes, det_boxes)):
        image_info = {"id": i, "file_name": f"image_{i}.jpg"}
        coco_gt_data["images"].append(image_info)
        coco_det_data["images"].append(image_info)

        for box in gt:
            x1, y1, x2, y2 = box["bbox"]
            width = x2 - x1
            height = y2 - y1
            annotation = {
                "id": annotation_id,
                "image_id": i,
                "category_id": box["label"],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
            coco_gt_data["annotations"].append(annotation)
            annotation_id += 1

        for box in det:
            x1, y1, x2, y2 = box["bbox"]
            width = x2 - x1
            height = y2 - y1
            annotation = {
                "id": annotation_id,
                "image_id": i,
                "category_id": box["label"],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "score": box["score"],
            }
            coco_det_data["annotations"].append(annotation)
            annotation_id += 1

    return coco_gt_data, coco_det_data


def main():
    config = read_yaml_file("/mnt/d/Workspace/FireSmokeDetection/ai_service/convert_json_and_metric_0.1.0/config/eval_config.yaml") 
    
    dataset_folder = config["ground_truth_dataset"]
    image_folder = os.path.join(dataset_folder, "images")

    # Load models
    model = YOLOv8(config["model"])

    # Load classes dict for mapping
    gt_cfg = read_yaml_file(config["ground_truth_cfg_path"])
    gt_classes = gt_cfg["names"]
    gt_cls_dict = {i: cls_name for i, cls_name in enumerate(gt_classes)}
    model_classes = config["model"]["classes"]
    model_cls_dict = {cls_name: i for i, cls_name in enumerate(model_classes)}

    gt_remap_dict = {i: model_cls_dict[cls_name] for i, cls_name in gt_cls_dict.items()}

    all_gt_boxes = []
    all_det_boxes = []
    # Duyệt qua từng ảnh và kiểm tra
    for image_name in os.listdir(image_folder):
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            continue

        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Warning: Unable to read image at {image_path}")
            continue

        gt_boxes = get_gt_boxes(image_path, gt_remap_dict)
        
        predictions = model.infer(img)
        det_boxes = [
            {
                "bbox": pred["xyxy"], 
                "label": pred["cls_id"], 
                "score": pred["conf"],
            }
            for pred in predictions
        ]  

        all_gt_boxes.append(gt_boxes)
        all_det_boxes.append(det_boxes)

    # Tạo COCO và gọi createIndex
    coco_gt_data, coco_det_data = create_coco_format(all_gt_boxes, all_det_boxes)

    # Tạo index cho COCO
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_data
    coco_gt.createIndex()

    coco_dt = COCO()
    coco_dt.dataset = coco_det_data
    coco_dt.createIndex()
   
    # Chạy đánh giá
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = range(len(all_gt_boxes))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # In ra các chỉ số
    print("Đánh giá hoàn tất:")
    print(f"mAP50-95:  {coco_eval.stats[0]:.2f}")
    print(f"Precision: {coco_eval.stats[1]:.2f}")
    print(f"Recall:    {coco_eval.stats[2]:.2f}")
    print(
        f"F1 Score:  {2 * (coco_eval.stats[1] * coco_eval.stats[2]) / (coco_eval.stats[1] + coco_eval.stats[2]):.2f}"
    )


if __name__ == "__main__":
    main()
