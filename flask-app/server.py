from flask import Flask, request
import base64
import os
import numpy as np
import cv2

from src.utils import (
  read_yaml_file,
  draw_box
)
from src.model import YOLOWrapper

app = Flask(__name__)

def init_model():
  config_file = os.path.abspath(r"./config/model.yaml")
  config = read_yaml_file(config_file)
  model = YOLOWrapper(config["model"])

  return model

model = init_model()

def image_from_base64(base64_string):
  nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  return img

def image_to_base64(image):
  _, buffer = cv2.imencode('.jpg', image)
  encoded_string = base64.b64encode(buffer).decode('utf-8')

  return encoded_string

@app.route("/model", methods = ["POST"])
def model_detection():
  data = request.json
  camera = data["camera"]

  image = image_from_base64(camera)

  predictions = model.infer(image)

  predictions_dict = {}

  for prediction in predictions:
    draw_box(image, prediction)

    class_id = prediction["cls_id"]
    if (class_id not in predictions_dict):
      predictions_dict[class_id] = 0
    predictions_dict[class_id] += 1

  base64_image = image_to_base64(image)

  res = {
    "total": len(predictions),
    "products": [],
    "processed": base64_image
  }

  for class_id in predictions_dict:
    count = predictions_dict[class_id]
    res["products"].append({ "id": int(class_id), "count": int(count) })

  return res

if __name__ == "__main__":
  app.run(debug=True,host='0.0.0.0')