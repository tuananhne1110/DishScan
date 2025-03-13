from flask import Flask, request
import base64
import sys

app = Flask(__name__)

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

  # TODO: model processing

  print('data', data, flush=True)

  return {
    "total": 3,
    "items": [
      { "id": 1, "count": 1 },
      { "id": 3, "count": 2 }
    ]
  }

if __name__ == "__main__":
  app.run(debug=True,host='0.0.0.0')