import cv2
import yaml
import numpy as np
from pycocotools.coco import COCO


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

def draw_boxes(img, gt_labels, det_boxes):
    for label in gt_labels:
        x1, y1, x2, y2 = label["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"GT: {label['label']} ",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

    for box in det_boxes:
        x1, y1, x2, y2 = box["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"DET: {box['label']} ",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    return img


def calculate_iou(box1, box2):
    np_box1 = np.array(box1, dtype=np.float64)
    np_box2 = np.array(box2, dtype=np.float64)
    x1, y1, x2, y2 = np_box1
    x1g, y1g, x2g, y2g = np_box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0


def xywh2xyxy(xywh):
    """
    Converts bounding box coordinates from (x_center, y_center, width, height) format to (x_min, y_min, x_max, y_max) format.

    Args:
        xywh (np.array) with shape (4,): A tuple containing (x_center, y_center, width, height) of the bounding box.

    Returns:
        xyxy (tuple(int)): xyxy location of the bounding box.
    """

    x_center, y_center, bbox_width, bbox_height = xywh
    x_min = int(x_center - bbox_width / 2)
    y_min = int(y_center - bbox_height / 2)
    x_max = int(x_center + bbox_width / 2)
    y_max = int(y_center + bbox_height / 2)

    return x_min, y_min, x_max, y_max


def draw_bbox(
    image, bbox, bbox_type="xyxy", bbox_color=(0, 0, 255), label=None
):
    """
    Draws a bounding box on the image.

    Args:
        image (np.ndarray or str): The image on which the bounding box will be drawn.
        bbox (list or np.ndarray): The bounding box coordinates. If it's a list, it should be in the format specified by bbox_type.
        bbox_type (str): Type of bounding box coordinates. Should be either "xyxy" or "xywh" or "s_xywh".
        bbox_color (tuple): Color of the bounding box in BGR format.
        label (str, optional): Label to be displayed alongside the bounding box.

    Returns:
        temp_image (np.ndarray): Image with the bounding box drawn.
    """
    if isinstance(image, str):
        temp_image = cv2.imread(image)
    else:
        temp_image = image.copy()

    image_height, image_width = temp_image.shape[:2]

    temp_bbox = bbox.copy()
    if isinstance(bbox, list):
        temp_bbox = np.array(bbox)

    if "s_" in bbox_type:
        temp_bbox *= np.array(
            [image_width, image_height, image_width, image_height]
        )

    temp_bbox = temp_bbox.astype(np.int64)

    if "xyxy" in bbox_type:
        x_min, y_min, x_max, y_max = temp_bbox
    elif "xywh" in bbox_type:
        x_min, y_min, x_max, y_max = xywh2xyxy(temp_bbox)
    else:
        raise ValueError("Invalid bounding box type")

    cv2.rectangle(temp_image, (x_min, y_min), (x_max, y_max), bbox_color, 2)
    if label is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        background_position = (x_min, y_min)
        background_end_position = (
            x_min + text_width,
            y_min - text_height - 5,
        )
        cv2.rectangle(
            temp_image,
            background_position,
            background_end_position,
            bbox_color,
            -1,
        )
        cv2.putText(
            temp_image,
            label,
            (x_min, y_min),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return temp_image


# Load ground truth from JSON file
def load_ground_truth(json_path):
    coco = COCO(json_path)
    ground_truth_dict = {}
    for img_id in coco.imgs:
        file_name = coco.imgs[img_id]["file_name"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        ground_truth_dict[file_name] = []
        for ann in annotations:
            x, y, width, height = ann["bbox"]
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + width)
            y2 = int(y + height)
            category_id = ann["category_id"] 
            ground_truth_dict[file_name].append(
                {"bbox": [x1, y1, x2, y2], "label": category_id}
            )
    return ground_truth_dict





