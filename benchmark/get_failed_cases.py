import os
import shutil

import cv2
import numpy as np
from kano.dataset_utils import YoloImage

from modules.utils import (
    draw_bbox, 
    read_yaml_file, 
    calculate_iou,
)
from modules.yolov8 import YOLOv8
from modules.failed_cases_enum import FailedCases, FailedCasesFolder


def get_failed_case(gt_boxes, det_boxes, iou_threshold):
    # check number of boxes
    if len(gt_boxes) != len(det_boxes):
        return FailedCases.WRONG_NUMBER_BOXES.value
    
    # check iou between each pair of boxes
    has_wrong_classes = False
    matched_det = list()
    for gt in gt_boxes:
        matched = False
        for i, det in enumerate(det_boxes):
            if i in matched_det:
                continue
            iou = calculate_iou(gt["xyxy"], det["xyxy"])
            if iou > iou_threshold:
                # 2 box khác class
                if gt["cls_name"] != det["cls_name"]:
                    has_wrong_classes = True

                matched = True
                matched_det.append(i)
                break
        
        if not matched:
            return FailedCases.LOW_IOU_BOXES.value

    # check if wrong classes
    if has_wrong_classes:
        return FailedCases.WRONG_CLASSES_BOXES.value
    
    return FailedCases.NOTHING.value


def init_folders(failed_folder, relabel_folder):
    wrong_number_folder = os.path.join(failed_folder, FailedCasesFolder.WRONG_NUMBER_BOXES.value)
    low_iou_folder = os.path.join(failed_folder, FailedCasesFolder.LOW_IOU_BOXES.value)
    wrong_classes_folder = os.path.join(failed_folder, FailedCasesFolder.WRONG_CLASSES_BOXES.value)

    os.makedirs(failed_folder, exist_ok=True)
    os.makedirs(wrong_number_folder, exist_ok=True)
    os.makedirs(low_iou_folder, exist_ok=True)
    os.makedirs(wrong_classes_folder, exist_ok=True)
    
    if relabel_folder is not None:
        relabel_wrong_number_folder = os.path.join(relabel_folder, FailedCasesFolder.WRONG_NUMBER_BOXES.value)
        relabel_low_iou_folder = os.path.join(relabel_folder, FailedCasesFolder.LOW_IOU_BOXES.value)
        relabel_wrong_classes_folder = os.path.join(relabel_folder, FailedCasesFolder.WRONG_CLASSES_BOXES.value)
        
        os.makedirs(relabel_folder, exist_ok=True)
        os.makedirs(relabel_wrong_number_folder, exist_ok=True)
        os.makedirs(relabel_low_iou_folder, exist_ok=True)
        os.makedirs(relabel_wrong_classes_folder, exist_ok=True)


def get_gt_boxes(image_path, gt_cls_dict):
    yolo_img = YoloImage(image_path)
    gt_boxes = [
        {
            "xyxy": np.array(l["xyxy"]),
            "cls_name": gt_cls_dict[l["class"]],
        }
        for l in yolo_img.get_labels()
    ]
    return gt_boxes  


def get_det_boxes(model, image, model_cls_dict):
    predictions = model.infer(image)
    det_boxes = [
        {
            "xyxy": np.array(pred["xyxy"]), 
            "cls_name": model_cls_dict[pred["cls_id"]], 
        }
        for pred in predictions
    ]  
    return det_boxes


def main():
    config = read_yaml_file("/mnt/d/Workspace/FireSmokeDetection/ai_service/convert_json_and_metric_0.1.0/config/eval_config.yaml") 

    dataset_folder = config["ground_truth_dataset"]
    image_folder = os.path.join(dataset_folder, "images")

    # Create visualize folders for failed cases and relabeling
    failed_folder = config["failed_cases"]["visual_folder"]
    relabel_folder = config["failed_cases"]["relabel_folder"]
    init_folders(failed_folder, relabel_folder)

    # Load models
    model = YOLOv8(config["model"])

    # Load classes dict for mapping
    gt_cfg = read_yaml_file(config["ground_truth_cfg_path"])
    gt_classes = gt_cfg["names"]
    gt_cls_dict = {i: cls_name for i, cls_name in enumerate(gt_classes)}

    model_classes = config["model"]["classes"]
    model_cls_dict = {i: cls_name for i, cls_name in enumerate(model_classes)}

    wrong_number_count = 0
    low_iou_count = 0
    wrong_classes_count = 0
    error_image_count = 0
    correct_image_count = 0

    # Duyệt qua từng ảnh và kiểm tra
    for image_name in os.listdir(image_folder):
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            continue

        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Warning: Unable to read image at {image_path}")
            error_image_count += 1
            continue
        
        gt_boxes = get_gt_boxes(image_path, gt_cls_dict)
        det_boxes = get_det_boxes(model, img, model_cls_dict)
        failed_case = get_failed_case(gt_boxes, det_boxes, config["failed_cases"]["iou_threshold"])

        if failed_case == FailedCases.WRONG_NUMBER_BOXES.value:
            wrong_number_count += 1
            case = FailedCasesFolder.WRONG_NUMBER_BOXES.value
        if failed_case == FailedCases.LOW_IOU_BOXES.value:
            low_iou_count += 1
            case = FailedCasesFolder.LOW_IOU_BOXES.value
        if failed_case == FailedCases.WRONG_CLASSES_BOXES.value:
            wrong_classes_count += 1
            case = FailedCasesFolder.WRONG_CLASSES_BOXES.value

        if failed_case != FailedCases.NOTHING.value:
            save_image = img.copy()
            for box in det_boxes:
                save_image = draw_bbox(save_image, box["xyxy"], bbox_color=(0, 255, 0), label="----" + box["cls_name"])
            for box in gt_boxes:
                save_image = draw_bbox(save_image, box["xyxy"], bbox_color=(0, 0, 255), label=box["cls_name"])
            save_path = os.path.join(failed_folder, case, image_name) 
            cv2.imwrite(save_path, save_image)
            if relabel_folder is not None:
                move_path = os.path.join(relabel_folder, case) 
                shutil.move(image_path, move_path)
        else:
            correct_image_count += 1

    total = wrong_number_count + low_iou_count + wrong_classes_count + error_image_count + correct_image_count
    print(f"Number of images with wrong number of bboxes:  {wrong_number_count} - percentage: {int(wrong_number_count/total*100)}%")
    print(f"Number of images with low iou of bboxes:       {low_iou_count} - percentage: {int(low_iou_count/total*100)}%")
    print(f"Number of images with wrong classes of bboxes: {wrong_classes_count} - percentage: {int(wrong_classes_count/total*100)}%")
    print(f"Number of images cannot read:                  {error_image_count} - percentage: {int(error_image_count/total*100)}%")
    print(f"Number of images with correct predictions:     {correct_image_count} - percentage: {int(correct_image_count/total*100)}%")


if __name__ == "__main__":
    main()
