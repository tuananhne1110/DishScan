from flask import Flask, request
import base64
import sys

from ..benchmark.modules.utils import (
  read_yaml_file
)
from ..benchmark.modules.model import YOLOv8

app = Flask(__name__)

def init_model():
  config_file = os.path.abspath(r"../benchmark/configs/model.yaml")
  config = read_yaml_file(config_file)
  model = YOLOv8(config["model"])

  return model

model = init_model()

def image_from_base64(base64_string):
  return base64.b64decode(base64_string)

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
  print('data', predictions, flush=True)

  predictions_dict = {}

  for prediction in predictions:
    class_id = prediction.cls_id
    if (class_id not in predictions_dict):
      predictions_dict[class_id] = 0
    predictions_dict[class_id] += 1

  res = {
    "total": len(predictions),
    "items": []
  }

  for class_id in predictions_dict:
    count = predictions_dict[class_id]
    res.items.append({ "id": class_id, count: count })

  return res

if __name__ == "__main__":
  app.run(debug=True,host='0.0.0.0')