# DishScan - YOLOv11 applied for pastry recognition

### Set up the dependencies

- All the required dependencies are listed in `requirements.txt` file, run the following command to install the dependencies using pip.

```sh
pip install -r requirements.txt
```

### Inference

- By default, the inference uses camera at index 0 as the input stream. You can configure this camera index by the variable `camera_index` in the `main.py` script.

- Run the main script via

```sh
python main.py
```

- The camera stream and YOLOv11 model takes about 30 seconds for initialization, after that, there will be a window represents the frame from the camera capture including the detected bounding boxes for bakery items. 