import unittest
from unittest.mock import MagicMock, patch
import cv2
import os
import shutil
from src.core import MangaTranslator

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure we have the test image
        if not os.path.exists("test.jpg"):
             # Create a dummy image if real one is missing,
             # but better to rely on existing one if possible
             img = cv2.imread("test.jpg")
             if img is None:
                 print("Creating dummy test.jpg")
                 img = np.zeros((500, 500, 3), dtype=np.uint8)
                 cv2.putText(img, "Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                 cv2.imwrite("test.jpg", img)

    def setUp(self):
        self.mock_yolo = MagicMock()
        self.mock_reader = MagicMock()
        self.mock_gemini = MagicMock()

        self.patcher_yolo = patch('src.core.YOLO', return_value=self.mock_yolo)
        self.patcher_easyocr = patch('src.core.easyocr.Reader', return_value=self.mock_reader)
        self.patcher_genai = patch('src.core.genai')

        self.mock_yolo_cls = self.patcher_yolo.start()
        self.mock_easyocr_cls = self.patcher_easyocr.start()
        self.mock_genai_mod = self.patcher_genai.start()

        self.mock_genai_mod.GenerativeModel.return_value = self.mock_gemini

    def tearDown(self):
        self.patcher_yolo.stop()
        self.patcher_easyocr.stop()
        self.patcher_genai.stop()

    def test_full_flow(self):
        """
        Tests the full flow with real image processing but mocked AI.
        """
        translator = MangaTranslator("best.pt", "dummy_key")

        # Mock YOLO to return a box around the center
        # We need to look at test.jpg dimensions or assume generic
        img = cv2.imread("test.jpg")
        h, w = img.shape[:2]

        # Define a box in the middle
        box = [w//4, h//4, 3*w//4, 3*h//4, 0.9, 0]

        mock_result = MagicMock()
        mock_result.boxes.data.tolist.return_value = [box]
        self.mock_yolo.return_value = [mock_result]

        # Mock OCR
        self.mock_reader.readtext.return_value = ["Manga", "Text"]

        # Mock Gemini
        mock_response = MagicMock()
        # Return a valid JSON
        mock_response.text = '[{"id": 0, "translated": "Manga Yazisi"}]'
        self.mock_gemini.generate_content.return_value = mock_response

        output_path = "integration_result.jpg"
        result = translator.translate_image("test.jpg", output_path=output_path)

        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output_path))

        # Verify the file is a valid image
        saved_img = cv2.imread(output_path)
        self.assertIsNotNone(saved_img)
        self.assertEqual(saved_img.shape, img.shape)

        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == '__main__':
    unittest.main()
