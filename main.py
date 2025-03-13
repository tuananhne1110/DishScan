import os
import shutil

import cv2
import numpy as np
from kano.dataset_utils import YoloImage

from benchmark.modules.utils import (
    draw_box,
    read_yaml_file, 
)
from benchmark.modules.model import YOLOv8

def init_camera():
    cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    return cam

def init_model():
    config_file = os.path.abspath(r"./benchmark/configs/eval_config.yaml")
    config = read_yaml_file(config_file) 
    model = YOLOv8(config["model"])

    return model


def main():
  data_yaml_file = os.path.abspath(r"./benchmark/configs/data.yaml")
  data_config = read_yaml_file(data_yaml_file)

  cam = init_camera()
  model = init_model()

  while(True):
    ret, frame = cam.read()

    predictions = model.infer(frame)

    if (len(predictions)):
      print('predictions', predictions)
      for prediction in predictions:
        draw_box(frame, prediction)

    cv2.imshow('cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cam.release()

  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
