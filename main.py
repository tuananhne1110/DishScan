import os
import cv2

from benchmark.modules.utils import (
    draw_box,
    read_yaml_file, 
)
from benchmark.modules.model import YOLOWrapper

def init_camera(camera_index):
    cam = cv2.VideoCapture(camera_index)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    return cam

def init_model():
    config_file = os.path.abspath(r"./benchmark/configs/model.yaml")
    config = read_yaml_file(config_file)
    model = YOLOWrapper(config["model"])

    return model

def main():
  camera_index = 0
  cam = init_camera(camera_index)
  
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
