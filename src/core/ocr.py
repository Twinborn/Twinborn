
import os
import numpy as np
from PIL import Image
from src.utils.logger import setup_logger

logger = setup_logger("OCR")

class OCRProcessor:
    def __init__(self):
        self.det_model = None
        self.det_processor = None
        self.rec_model = None
        self.rec_processor = None
        self._load_models()

    def _load_models(self):
        try:
            from surya.model.detection.segformer import load_model as load_det_model
            from surya.model.detection.segformer import load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor

            logger.info("Loading Surya OCR models...")
            self.det_model = load_det_model()
            self.det_processor = load_det_processor()
            self.rec_model = load_rec_model()
            self.rec_processor = load_rec_processor()
            logger.info("Surya OCR models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Surya OCR models: {e}")
            raise

    def extract_text(self, image, language="en"):
        """
        Extract text from an image region (numpy array or PIL Image).
        """
        try:
            from surya.ocr import run_ocr

            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Use 'en' as default for Surya if language is not specific supported code
            # Surya supports many langs but let's stick to common ones
            langs = [language] if language else ["en"]

            predictions = run_ocr(
                [image],
                langs,
                self.det_model,
                self.det_processor,
                self.rec_model,
                self.rec_processor
            )

            if not predictions or len(predictions) == 0:
                return None

            text_lines = predictions[0].text_lines
            full_text = " ".join([line.text for line in text_lines])
            return full_text.strip()

        except Exception as e:
            logger.error(f"OCR Extraction failed: {e}")
            return None
