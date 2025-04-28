import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import screeninfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CAMERA_INDEX = 0
DEFAULT_OUTPUT_DIR = Path("images")
SPACE_KEY = 32
ESC_KEY = 27
WINDOW_NAME = "Camera Capture"


def get_screen_resolution() -> Tuple[int, int]:
    """Get the resolution of the primary monitor.
    
    Returns:
        Tuple[int, int]: Screen width and height in pixels.
    """
    try:
        monitor = screeninfo.get_monitors()[0]
        return monitor.width, monitor.height
    except IndexError as e:
        raise RuntimeError("No monitors detected") from e


def initialize_camera(camera_index: int, resolution: Tuple[int, int]) -> cv2.VideoCapture:
    """Initialize and configure the camera device.
    
    Args:
        camera_index: Index of the camera device
        resolution: Tuple of (width, height) for desired resolution
    
    Returns:
        cv2.VideoCapture: Configured camera object
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to initialize camera at index {camera_index}")

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Disable auto-focus 
    if cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        logger.info("Auto-focus disabled")
    else:
        logger.warning("Auto-focus control not supported for this camera")

    # Set manual focus to a desired value (range depend on the camera)
    focus_value = 10
    if cap.set(cv2.CAP_PROP_FOCUS, focus_value):
        logger.info(f"Manual focus set to {focus_value}")
    else:
        logger.warning("Manual focus control not supported on this camera.")
        
    # Verify actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_width, actual_height) != resolution:
        logger.warning(
            f"Camera resolution adjusted to {actual_width}x{actual_height} "
            f"(requested {resolution[0]}x{resolution[1]})"
        )

    return cap


def get_next_image_number(output_dir: Path) -> int:
    """Get the next sequential image number in the output directory.
    
    Args:
        output_dir: Directory to search for existing images
    
    Returns:
        int: Next available image number
    """
    existing_images = list(output_dir.glob("captured_image_*.jpg"))
    if not existing_images:
        return 1

    numbers = []
    for img_path in existing_images:
        try:
            number = int(img_path.stem.split("_")[-1])
            numbers.append(number)
        except (ValueError, IndexError):
            continue

    return max(numbers) + 1 if numbers else 1


def capture_images(
    output_dir: Path,
    camera_index: int = DEFAULT_CAMERA_INDEX,
    window_name: str = WINDOW_NAME,
) -> None:
    """Main image capture loop.
    
    Args:
        output_dir: Directory to save captured images
        camera_index: Index of the camera device
        window_name: Name of the display window
    """
    resolution = get_screen_resolution()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_number = get_next_image_number(output_dir)

    try:
        cap = initialize_camera(camera_index, resolution)
        logger.info("Camera initialized. Press SPACE to capture, ESC to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == SPACE_KEY:
                image_path = output_dir / f"captured_image_{image_number:04d}.jpg"
                cv2.imwrite(str(image_path), frame)
                logger.info(f"Image saved: {image_path}")
                image_number += 1

            if key == ESC_KEY:
                logger.info("Exiting...")
                break

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera resources released")


def parse_arguments():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Camera image capture application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save captured images",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="Index of camera device to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    capture_images(
        output_dir=args.output_dir,
        camera_index=args.camera_index,
    )