
import os
import cv2
import numpy as np
from PIL import Image
from src.utils.logger import setup_logger

logger = setup_logger("Detector")

class BubbleDetector:
    def __init__(self, model_path="model.pt"):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if os.path.exists(self.model_path):
                logger.info(f"Loading YOLO model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, image):
        """
        Detect bubbles, frames, and text in the image.
        Returns: List of detections [x1, y1, x2, y2, conf, class_id]
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image

            results = self.model(img_array)
            detections = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    detections.append([x1, y1, x2, y2, conf, cls])

            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
