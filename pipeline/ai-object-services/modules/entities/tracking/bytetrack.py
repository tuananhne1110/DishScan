from typing import List, Dict, Any, Optional
import logging
from boxmot import ByteTrack
from configs.loader import cfg

# Set up logging
logger = logging.getLogger(__name__)


class ByteTracker:
    """A class to handle object tracking using ByteTrack.

    Attributes:
        tracker (ByteTrack): The ByteTrack tracker instance.
    """

    def __init__(self):
        """Initialize the ByteTracker and load the tracker configuration."""
        self.tracker: Optional[ByteTrack] = None
        self.load_tracker()

    def load_tracker(self) -> None:
        """Load and configure the ByteTrack tracker.

        Raises:
            RuntimeError: If the tracker fails to initialize.
        """
        try:
            self.tracker = ByteTrack(
                track_thresh=cfg["tracker"]["track_thresh"],
                match_thresh=cfg["tracker"]["match_thresh"],
                track_buffer=cfg["tracker"]["track_buffer"],  # Fixed typo: trackW_buffer -> track_buffer
                frame_rate=cfg["tracker"]["frame_rate"],
            )
            logger.info("ByteTrack tracker initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ByteTrack tracker: {e}", exc_info=True)
            raise RuntimeError(f"Tracker initialization failed: {e}")

    def update(self, predictions: List[Any], image: Any) -> List[Dict[str, Any]]:
        """Update the tracker with new predictions and an image frame.

        Args:
            predictions: A list of detection predictions.
            image: The current image frame.

        Returns:
            A list of dictionaries containing tracked object information (class, confidence, bbox, ID).
        """
        if self.tracker is None:
            logger.error("Tracker is not initialized.")
            return []

        try:
            # Update the tracker with new predictions
            tracked_objects = self.tracker.update(predictions, image)
            outputs = []

            if len(tracked_objects) > 0:
                # Extract object attributes
                xyxys = tracked_objects[:, 0:4].astype("int").tolist()
                ids = tracked_objects[:, 4].astype("int").tolist()
                confs = tracked_objects[:, 5].tolist()
                classes = tracked_objects[:, 6].astype("int").tolist()

                # Format the outputs
                for xyxy, obj_id, conf, cls in zip(xyxys, ids, confs, classes):
                    outputs.append({
                        "cls": cls,
                        "conf": conf,
                        "bbox": xyxy,
                        "id": obj_id,
                    })

            return outputs

        except Exception as e:
            logger.error(f"Failed to update tracker: {e}", exc_info=True)
            return []