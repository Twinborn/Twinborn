import logging
from PIL import Image
import numpy as np
import cv2
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
# Import FoundationPredictor if needed, but it seems RecognitionPredictor needs an instance of it now.
# Based on the error: RecognitionPredictor.__init__() missing 1 required positional argument: 'foundation_predictor'
# And the help: RecognitionPredictor(foundation_predictor: 'FoundationPredictor')
from surya.recognition import FoundationPredictor

logger = logging.getLogger(__name__)

class OCRProcessor:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OCRProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Singleton initialization of Surya models to avoid reloading.
        """
        if self._initialized:
            return

        try:
            logger.info("Loading Surya OCR models...")
            # Load detection models
            self.det_predictor = DetectionPredictor()

            # Load recognition models
            # New Surya API requires initializing a FoundationPredictor first it seems?
            # Or is RecognitionPredictor just a wrapper?
            # The help says: class RecognitionPredictor(surya.common.predictor.BasePredictor)
            # init(self, foundation_predictor: 'FoundationPredictor')

            # So we must create a FoundationPredictor.
            self.foundation_predictor = FoundationPredictor()
            self.rec_predictor = RecognitionPredictor(self.foundation_predictor)

            self._initialized = True
            logger.info("Surya OCR models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Surya OCR models: {e}")
            raise

    def process_bubble(self, image: np.ndarray, bubble_data: dict, languages: list = None) -> str:
        """
        Runs OCR on a bubble. Uses internal text crops if available for better accuracy.

        Args:
            image: Full original image (numpy array BGR).
            bubble_data: Dictionary containing 'bbox' and 'text_crops'.
            languages: List of languages for Surya (default ['en']).

        Returns:
            Extracted text string.
        """
        if languages is None:
            languages = ["en"]

        try:
            bx1, by1, bx2, by2 = bubble_data['bbox']

            # If we have specific text crops detected by YOLO, use them!
            text_regions = bubble_data.get('text_crops', [])

            extracted_lines = []

            if text_regions:
                text_regions.sort(key=lambda x: x['bbox'][1])

                for text_item in text_regions:
                    tx1, ty1, tx2, ty2 = text_item['bbox']

                    pad = 5
                    h, w = image.shape[:2]
                    tx1 = max(0, tx1 - pad)
                    ty1 = max(0, ty1 - pad)
                    tx2 = min(w, tx2 + pad)
                    ty2 = min(h, ty2 + pad)

                    roi = image[ty1:ty2, tx1:tx2]
                    if roi.size == 0: continue

                    text = self._run_surya_on_roi(roi, languages)
                    if text:
                        extracted_lines.append(text)
            else:
                # Fallback: OCR the entire bubble if no specific text detected
                roi = image[by1:by2, bx1:bx2]
                if roi.size > 0:
                    text = self._run_surya_on_roi(roi, languages)
                    if text:
                        extracted_lines.append(text)

            return " ".join(extracted_lines)

        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return ""

    def _run_surya_on_roi(self, roi: np.ndarray, languages: list) -> str:
        """Helper to run Surya on a specific image crop."""
        try:
            # OpenCV BGR -> PIL RGB
            pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

            # 1. Detect text lines
            # det_predictor calls return list of TextDetectionResult
            det_predictions = self.det_predictor([pil_image])

            # 2. Recognize text
            # rec_predictor(images, task_names, det_predictor=...)
            # We can pass the det_predictor instance or the results?
            # Help says: __call__(self, images: 'List[Image.Image]', task_names: 'List[str] | None' = None, det_predictor: 'DetectionPredictor | None' = None, ...)

            # It seems we can just run recognition and it might handle detection internally if we pass the predictor?
            # Or we can pass 'polygons' from det_predictions?

            # Let's try passing det_predictor instance to it, as the signature suggests.
            # "det_predictor: 'DetectionPredictor | None' = None"

            rec_preds = self.rec_predictor([pil_image], [languages], det_predictor=self.det_predictor)

            if not rec_preds or not rec_preds[0].text_lines:
                return ""

            full_text = " ".join([line.text for line in rec_preds[0].text_lines])
            return full_text.strip()

        except Exception as e:
            logger.error(f"Surya low-level error: {e}")
            return ""
