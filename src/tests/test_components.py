
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.core.image_processing import ImageProcessor
from src.core.translator import Translator

class TestImageProcessor(unittest.TestCase):
    def test_sort_bubbles(self):
        # [x1, y1, x2, y2, conf, cls]
        # b1: top right
        b1 = [100, 10, 150, 60, 0.9, 0]
        # b2: top left
        b2 = [10, 10, 60, 60, 0.9, 0]
        # b3: bottom
        b3 = [50, 200, 100, 250, 0.9, 0]

        bubbles = [b1, b2, b3]

        sorted_bubbles = ImageProcessor.sort_bubbles(bubbles)

        # Expect: b1 (top right), b2 (top left), b3 (bottom)

        self.assertEqual(sorted_bubbles[0], b1)
        self.assertEqual(sorted_bubbles[1], b2)
        self.assertEqual(sorted_bubbles[2], b3)

class TestTranslator(unittest.TestCase):
    @patch('src.core.translator.genai')
    def test_translate(self, mock_genai):
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        mock_response = MagicMock()
        mock_response.text = "Merhaba Dünya"
        mock_model.generate_content.return_value = mock_response

        translator = Translator(api_key="fake_key")
        result = translator.translate("Hello World", "Turkish")

        self.assertEqual(result, "Merhaba Dünya")

if __name__ == '__main__':
    unittest.main()
