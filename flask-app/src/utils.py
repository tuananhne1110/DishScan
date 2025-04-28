import yaml

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def draw_box(img, box):
    colors = [
        (255,0,0),
        (255,255,0),
        (0,0,0),
        (255,0,255),
        (0,255,0),
        (0,255,255),
        (0,0,255)
    ]

    x1, y1, x2, y2 = box["xyxy"]
    class_id = box["cls_id"]
    color = colors[class_id % len(colors)]

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = box["cls_name"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_w, text_h = text_size
    padding = 5

    cv2.rectangle(img, (x1, y1 - text_h - padding), (x1 + text_w + padding, y1), color, -1)
    cv2.putText(
        img,
        label,
        (x1, y1 - 5),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )