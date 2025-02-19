# File: modules/controllers/fire_controller.py
import os
import json
import time
import numpy as np  # Add this import

import cv2
import simplejpeg

from configs.loader import cfg
from logs.log_handler import logger
from modules.entities.detection.yolov8.pt.model import YOLOv8Detection
from modules.entities.redis.redis_connection import RedisConnection

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class FireController:
    def __init__(self, queue_fire):
        self.redis_conn = RedisConnection()
        self.fire_detector = YOLOv8Detection()
        self.queue_metadata = queue_fire

    def get_last_elements(self):
        data = self.queue_metadata.get()
        return data

    def run(self):
        logger.debug(f"Fire Controller PID: {os.getpid()}")
        while True:
            last_elements = self.get_last_elements()
            if last_elements is None:
                continue

            camera_id = last_elements.get("camera_id", "unknown")
            timestamp = last_elements.get("timestamp", "")
            frame = last_elements["metadata"].get("frame")
            if frame is None:
                logger.error("No frame data found in last_elements!")
                continue

            # Use __call__ method to get processed detections
            detections = self.fire_detector(frame)  # This returns list of dicts
            logger.debug(f"Camera_id: {camera_id}")
            logger.debug(f"Inference Results: {detections}")

            metadata = {
                "camera_id": camera_id,
                "timestamp": timestamp,
                "objects": detections,
            }
            frame_bytes = simplejpeg.encode_jpeg(
                frame,
                quality=cfg["common"]["redis"]["jpeg_quality"],
                colorspace=cfg["common"]["redis"]["colorspace"],
            )
            msg = {
                "metadata": json.dumps(metadata, default=convert_numpy_types),  # Handle numpy types
                "frame": frame_bytes,
            }
            topic = f"ai_service:{camera_id}"
            self.redis_conn.send_message(msg, topic)