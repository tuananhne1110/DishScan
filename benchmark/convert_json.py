import os
import json

from PIL import Image

from modules.utils import read_yaml_file


def yolo2coco(yolo_dataset, categories_dict, yolo_cls_dict):
    categories = list()
    for cat, i in categories_dict.items():
        categories.append({"id": i, "name": cat, "supercategory": "none"})

    coco_data = {
        "info": {
            "year": "2024",
            "version": "1.0",
            "description": "",
            "contributor": "",
            "url": "",
            "date_created": "2024-11-01T08:01:46+00:00",
        },
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "name": "CC BY 4.0",
            }
        ],
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    annotation_id = 0
    image_id = 0

    label_folder = os.path.join(yolo_dataset, "labels")
    image_folder = os.path.join(yolo_dataset, "images")


    for label_file_name in os.listdir(label_folder):
        if not label_file_name.lower().endswith(".txt"):
            continue  # Bỏ qua các tệp không phải là file txt

        label_file_path = os.path.join(label_folder, label_file_name)

        image_name = label_file_name.replace(".txt", ".jpg")
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_name} not found.")
            continue

        img = Image.open(image_path)
        img_width, img_height = img.size

        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": image_name,
                "height": img_height,
                "width": img_width,
                "date_captured": "2024-11-01T08:01:46+00:00",
            }
        )

        with open(label_file_path, "r") as label_file:
            # Duyệt qua từng dòng trong file nhãn
            for line in label_file:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                class_name = yolo_cls_dict[class_id]
                class_id = categories_dict[class_name]

                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                box_width = width * img_width
                box_height = height * img_height

                # Tạo annotation cho từng bounding box
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }

                coco_data["annotations"].append(annotation)
                annotation_id += 1

        image_id += 1

    output_json_path = os.path.join(image_folder, "_annotations.coco.json")
    with open(output_json_path, "w") as json_file:
        json.dump(coco_data, json_file, indent=4)
    print(f"COCO format JSON saved to {output_json_path}")


def main():
    cfg = read_yaml_file("config/convert_config.yaml")

    yolo_cfg = read_yaml_file(cfg["yolo_cfg_file"])

    if not all(item in cfg["categories"] for item in yolo_cfg["names"]):
        raise ValueError("The categories must include all classes of the YOLO dataset")
    
    yolo_cls_dict = {i: cls_name for i, cls_name in enumerate(yolo_cfg["names"])}
    categories_dict = {cat: i for i, cat in enumerate(cfg["categories"])}

    yolo2coco(cfg["yolo_dataset"], categories_dict, yolo_cls_dict)



if __name__ == "__main__":
    main()
