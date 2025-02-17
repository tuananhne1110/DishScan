import cv2
import torch
import logging
from typing import List, Dict, Optional
from ultralytics import YOLO
from configs.loader import cfg

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOv8Detector:
    """YOLOv8 object detection model for performing inference and visualization.
    
    Attributes:
        model (YOLO): Loaded YOLOv8 model
        confidence_threshold (float): Minimum confidence threshold for detections
        iou_threshold (float): Intersection over Union threshold for NMS
        device (str): Computation device (cuda/cpu)
        class_names (Dict[int, str]): Mapping of class IDs to names
    """

    def __init__(
        self, 
        model_path: str = cfg["model"]["detection"]["yolov8"]["bread"]["model_path"]["pt"],
        confidence_threshold: float = cfg["model"]["detection"]["yolov8"]["confidence_threshold"],
        iou_threshold: float = cfg["model"]["detection"]["yolov8"]["iou_threshold"],
        device: Optional[str] = None,
    ):
        """Initialize YOLOv8 detector with configuration parameters."""
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8 model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = self.model.names

    def infer(self, frame: torch.Tensor, target_classes: Optional[List[int]] = None) -> List[Dict]:
        """Perform object detection on input frame.
        
        Args:
            frame: Input image tensor (HWC format)
            target_classes: List of class IDs to filter (None for all classes)
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class info
        """
        if frame is None or frame.size == 0:
            logger.warning("Received empty input frame")
            return []

        try:
            with torch.no_grad():
                results = self.model.predict(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False
                )

            if not results:
                return []

            detections = []
            for result in results[0].boxes:
                x1, y1, x2, y2 = result.xyxy[0].tolist()[:4]
                conf = result.conf.item()
                cls_id = int(result.cls.item())

                if target_classes is None or cls_id in target_classes:
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": self.class_names.get(cls_id, "unknown")
                    })

            return detections

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []

    def draw_box(
        self,
        frame: torch.Tensor,
        detections: List[Dict],
        color: Optional[tuple] = None,
        show_labels: bool = True
    ) -> torch.Tensor:
        """Visualize detections on input frame.
        
        Args:
            frame: Input image tensor
            detections: List of detections from infer()
            color: Optional BGR color tuple
            show_labels: Whether to display class labels
            
        Returns:
            Image with visualized detections
        """
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls_id = det["class_id"]
            label = f"{det['class_name']} {det['confidence']:.2f}"

            # Generate color based on class ID if not specified
            bgr_color = color or (int(255 * (cls_id/len(self.class_names))), 
                                 int(255 * (1 - cls_id/len(self.class_names))), 
                                 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)

            if show_labels:
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 4),
                    (x1 + text_width, y1),
                    bgr_color,
                    -1
                )
                # Draw text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        return frame