
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Create dummy font
        if not os.path.exists("fonts"):
            os.makedirs("fonts")

    @patch('src.core.detector.BubbleDetector._load_model')
    @patch('src.core.ocr.OCRProcessor._load_models')
    def test_pipeline_initialization(self, mock_load_ocr, mock_load_det):
        # Mock methods to avoid real loading
        mock_load_det.return_value = None
        mock_load_ocr.return_value = None

        from src.app import MangaProcessor

        processor = MangaProcessor()
        # Mock the internal objects since _load_model was skipped
        processor.detector.model = MagicMock()

        self.assertIsNotNone(processor.detector)
        self.assertIsNotNone(processor.ocr)
        self.assertIsNotNone(processor.translator)

if __name__ == '__main__':
    unittest.main()
