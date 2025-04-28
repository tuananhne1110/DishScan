from flask import Flask, request
import base64
import os
import numpy as np
import cv2

from src.utils import (
  read_yaml_file
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

@app.route("/model", methods = ["POST"])
def model_detection():
  data = request.json
  camera = data["camera"]

  image = image_from_base64(camera)

  predictions = model.infer(image)

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