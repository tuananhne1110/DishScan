from flask import Flask, request
import base64
import sys
import os
import numpy as np
import cv2

from src.utils import (
  read_yaml_file
)
from src.model import YOLOv8

app = Flask(__name__)

def init_model():
  config_file = os.path.abspath(r"./config/model.yaml")
  config = read_yaml_file(config_file)
  model = YOLOv8(config["model"])

  return model

model = init_model()

def image_from_base64(base64_string):
  nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  return img

@app.route("/model", methods = ["POST"])
def model_detection():
  data = request.json
  cam1 = data["camera1"]
  cam2 = data["camera2"]
  cam3 = data["camera3"]

  image1 = image_from_base64(cam1)
  image2 = image_from_base64(cam2)
  image3 = image_from_base64(cam3)

  predictions = model.infer(image2)

  predictions_dict = {}

  for prediction in predictions:
    class_id = prediction["cls_id"]
    if (class_id not in predictions_dict):
      predictions_dict[class_id] = 0
    predictions_dict[class_id] += 1

  res = {
    "total": len(predictions),
    "products": []
  }

  for class_id in predictions_dict:
    count = predictions_dict[class_id]
    res["products"].append({ "id": int(class_id), "count": int(count) })

  return res

if __name__ == "__main__":
  app.run(debug=True,host='0.0.0.0')