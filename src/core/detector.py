import logging
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class Detector:
    def __init__(self, model_path: str):
        """
        Initializes the YOLO detector.

        Args:
            model_path: Path to the .pt model file.
        """
        self.model = YOLO(model_path)
        # Class mapping based on user description
        # 0: bubble, 1: frame, 2: text
        self.CLASS_BUBBLE = 0
        self.CLASS_FRAME = 1
        self.CLASS_TEXT = 2

    def detect(self, image, conf_threshold=0.25, iou_threshold=0.45) -> List[Dict]:
        """
        Detects bubbles and text in the image and associates text with bubbles.

        Args:
            image: Input image (numpy array, PIL image, or path).
            conf_threshold: Confidence threshold for YOLO.
            iou_threshold: IOU threshold for NMS.

        Returns:
            List of dictionaries, where each dictionary represents a bubble:
            {
                'bbox': [x1, y1, x2, y2], (Bubble coordinates)
                'conf': float,
                'class_id': int,
                'text_crops': [ # List of text boxes inside this bubble
                    {
                        'bbox': [x1, y1, x2, y2], (Text coordinates)
                        'conf': float
                    },
                    ...
                ]
            }
        """
        results = self.model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)

        # We process the first result (single image mode)
        boxes = results[0].boxes.data.tolist()

        bubbles = []
        texts = []

        # 1. Separate Bubbles and Texts
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box
            cls_id = int(cls_id)

            item = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'conf': conf,
                'class_id': cls_id
            }

            if cls_id == self.CLASS_BUBBLE:
                item['text_crops'] = [] # Initialize empty list for children
                bubbles.append(item)
            elif cls_id == self.CLASS_TEXT:
                texts.append(item)
            # We ignore Frames (Class 1) for now as requested

        # 2. Associate Texts with Bubbles
        # A text is inside a bubble if its center point is inside the bubble box
        for text in texts:
            tx1, ty1, tx2, ty2 = text['bbox']
            t_center_x = (tx1 + tx2) / 2
            t_center_y = (ty1 + ty2) / 2

            matched = False
            for bubble in bubbles:
                bx1, by1, bx2, by2 = bubble['bbox']

                if bx1 <= t_center_x <= bx2 and by1 <= t_center_y <= by2:
                    bubble['text_crops'].append(text)
                    matched = True
                    break # Assuming text belongs to one bubble primarily

            # Note: If text is orphaned (no bubble), we currently ignore it.
            # In the future, we might want to treat orphaned text as its own bubble.

        return bubbles
