import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
import os
from src.core import MangaTranslator

class TestMangaTranslator(unittest.TestCase):
    def setUp(self):
        self.mock_yolo = MagicMock()
        self.mock_reader = MagicMock()
        self.mock_gemini = MagicMock()

        # Patch the external libraries
        self.patcher_yolo = patch('src.core.YOLO', return_value=self.mock_yolo)
        self.patcher_easyocr = patch('src.core.easyocr.Reader', return_value=self.mock_reader)
        self.patcher_genai = patch('src.core.genai')

        self.mock_yolo_cls = self.patcher_yolo.start()
        self.mock_easyocr_cls = self.patcher_easyocr.start()
        self.mock_genai_mod = self.patcher_genai.start()

        # Setup mock behavior
        self.mock_genai_mod.GenerativeModel.return_value = self.mock_gemini

    def tearDown(self):
        self.patcher_yolo.stop()
        self.patcher_easyocr.stop()
        self.patcher_genai.stop()

    def test_init(self):
        translator = MangaTranslator("dummy_model.pt", "dummy_key")
        self.assertIsNotNone(translator.yolo)
        self.assertIsNotNone(translator.reader)
        self.assertIsNotNone(translator.gemini)

    def test_process_bubble_white(self):
        translator = MangaTranslator("dummy_model.pt", "dummy_key")
        # Create a white image
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Add a black circle in middle
        cv2.circle(img, (50, 50), 20, (0, 0, 0), -1)

        processed, contour = translator.process_bubble(img.copy())

        # The black circle should be filled with white if logic works,
        # or at least the function shouldn't crash
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, img.shape)

    def test_translate_image_no_balloons(self):
        translator = MangaTranslator("dummy_model.pt", "dummy_key")

        # Mock YOLO results to return empty
        mock_result = MagicMock()
        mock_result.boxes.data.tolist.return_value = []
        self.mock_yolo.return_value = [mock_result]

        # Create dummy image
        img_path = "dummy_test.jpg"
        cv2.imwrite(img_path, np.zeros((100, 100, 3), dtype=np.uint8))

        try:
            result = translator.translate_image(img_path)
            self.assertIsNotNone(result)
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

    def test_translate_image_with_balloons(self):
        translator = MangaTranslator("dummy_model.pt", "dummy_key")

        # Mock YOLO results
        # [x1, y1, x2, y2, conf, cls]
        mock_result = MagicMock()
        mock_result.boxes.data.tolist.return_value = [[10, 10, 50, 50, 0.9, 0]]
        self.mock_yolo.return_value = [mock_result]

        # Mock OCR
        self.mock_reader.readtext.return_value = ["Hello", "World"]

        # Mock Gemini
        mock_response = MagicMock()
        mock_response.text = '[{"id": 0, "translated": "Merhaba Dunya"}]'
        self.mock_gemini.generate_content.return_value = mock_response

        # Create dummy image
        img_path = "dummy_test_2.jpg"
        cv2.imwrite(img_path, np.full((100, 100, 3), 255, dtype=np.uint8))

        try:
            result = translator.translate_image(img_path)
            self.assertIsNotNone(result)
            # Verify dependencies were called
            self.mock_yolo.assert_called()
            self.mock_reader.readtext.assert_called()
            self.mock_gemini.generate_content.assert_called()
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

if __name__ == '__main__':
    unittest.main()
