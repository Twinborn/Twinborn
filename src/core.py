import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import json
import textwrap
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MangaTranslator:
    def __init__(self, model_path, gemini_api_key, gemini_model_name='gemini-2.5-pro', font_path=None, ocr_scale=1.2, target_lang="Turkish"):
        self.model_path = model_path
        self.gemini_api_key = gemini_api_key
        self.gemini_model_name = gemini_model_name
        self.font_path = font_path
        self.ocr_scale = ocr_scale
        self.target_lang = target_lang

        self.yolo = None
        self.reader = None
        self.gemini = None

        self._load_models()

    def _load_models(self):
        try:
            logger.info("Loading YOLO model...")
            self.yolo = YOLO(self.model_path)

            logger.info("Loading EasyOCR...")
            self.reader = easyocr.Reader(['en'], gpu=True)

            logger.info("Configuring Gemini...")
            genai.configure(api_key=self.gemini_api_key)
            self.gemini = genai.GenerativeModel(self.gemini_model_name)

            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _manga_sort(self, box):
        x1, y1 = int(box[0]), int(box[1])
        row_id = int(y1 / 100)
        return (row_id, -x1)

    def _prepare_ocr_image(self, img):
        if self.ocr_scale == 1.0: return img
        h, w = img.shape[:2]
        new_h, new_w = int(h * self.ocr_scale), int(w * self.ocr_scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def process_bubble(self, image):
        """
        Finds the brightest area (bubble) and fills it with white.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = None
        if contours:
            try:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
                image[mask == 255] = (255, 255, 255)
            except:
                image[:] = (255, 255, 255)
        else:
            image[:] = (255, 255, 255)

        return image, largest_contour

    def add_text(self, image, text, bubble_contour):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        img_h, img_w = image.shape[:2]

        if bubble_contour is not None:
            x, y, w, h = cv2.boundingRect(bubble_contour)
        else:
            x, y, w, h = 0, 0, img_w, img_h

        if w < 10 or h < 10:
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)

        line_height = 16
        font_size = 14
        wrapping_ratio = 0.8

        # Default font fallback
        font = ImageFont.load_default()
        if self.font_path and os.path.exists(self.font_path):
             try:
                 font = ImageFont.truetype(self.font_path, size=font_size)
             except Exception as e:
                 logger.warning(f"Could not load font {self.font_path}, using default. Error: {e}")

        # Fitting Loop
        while True:
            if self.font_path and os.path.exists(self.font_path):
                try: font = ImageFont.truetype(self.font_path, size=font_size)
                except: font = ImageFont.load_default()

            target_width_chars = int((w * wrapping_ratio) / (font_size * 0.5))
            if target_width_chars < 1: target_width_chars = 1

            wrapped_text = textwrap.fill(text, width=target_width_chars)
            lines = wrapped_text.split('\n')
            total_text_height = len(lines) * line_height

            if (total_text_height > h or any(self._get_text_width(draw, l, font) > w for l in lines)) and font_size > 8:
                line_height -= 2
                font_size -= 2
                wrapping_ratio += 0.05
            else:
                break

        text_y = y + (h - total_text_height) // 2

        for line in lines:
            text_w = self._get_text_width(draw, line, font)
            text_x = x + (w - text_w) / 2
            draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))
            text_y += line_height

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)

    def _get_text_width(self, draw, text, font):
        try:
            if hasattr(draw, 'textlength'):
                return draw.textlength(text, font=font)
            else:
                return font.getsize(text)[0]
        except:
            return len(text) * 10 # Rough estimate

    def translate_image(self, image_path, output_path=None):
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None

        img_cv2 = cv2.imread(image_path)

        logger.info("Scanning for balloons...")
        results = self.yolo(img_cv2, verbose=False)
        boxes_data = results[0].boxes.data.tolist()

        if not boxes_data:
            logger.warning("No balloons found.")
            return img_cv2

        boxes_data.sort(key=self._manga_sort)
        logger.info(f"Found {len(boxes_data)} balloons.")

        havuz = []

        for i, box in enumerate(boxes_data):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            if (x2-x1) < 15: continue

            roi = img_cv2[y1:y2, x1:x2]
            roi_ocr = self._prepare_ocr_image(roi)

            try:
                res = self.reader.readtext(roi_ocr, detail=0, paragraph=True)
                if res:
                    text = " ".join(res)
                    havuz.append({"id": i, "original": text})
            except Exception as e:
                 logger.warning(f"OCR error on box {i}: {e}")
                 continue

        if not havuz:
            logger.warning("No text detected.")
            return img_cv2

        logger.info("Translating text...")
        try:
            prompt = f"""
            Translate manga text to {self.target_lang} (Casual style). Make it sound natural and fit for a manga context.
            Return valid JSON List ONLY. No extra text.
            Input: {json.dumps(havuz, ensure_ascii=False)}
            Output format: [ {{"id": 0, "translated": "..."}} ]
            """
            response = self.gemini.generate_content(prompt)

            clean = response.text.strip()
            if "```" in clean: clean = clean.split("```json")[-1].split("```")[0]
            if "[" in clean:
                start = clean.find("[")
                end = clean.rfind("]") + 1
                clean = clean[start:end]
                ceviri_havuzu = json.loads(clean)
            else:
                 ceviri_havuzu = []
        except Exception as e:
            logger.error(f"Translation error: {e}")
            ceviri_havuzu = [{"id": h["id"], "translated": h["original"]} for h in havuz]

        ceviri_dict = {item["id"]: item.get("translated", "") for item in ceviri_havuzu}

        logger.info("Rendering translated text...")
        for i, box in enumerate(boxes_data):
            if i not in ceviri_dict: continue

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            tr_text = ceviri_dict[i]

            bubble_img = img_cv2[y1:y2, x1:x2]
            cleaned_bubble, contour = self.process_bubble(bubble_img)
            final_bubble = self.add_text(cleaned_bubble, tr_text, contour)

            try:
                h_orig, w_orig = img_cv2[y1:y2, x1:x2].shape[:2]
                if final_bubble.shape[:2] != (h_orig, w_orig):
                    final_bubble = cv2.resize(final_bubble, (w_orig, h_orig))
                img_cv2[y1:y2, x1:x2] = final_bubble
            except: pass

        if output_path:
            cv2.imwrite(output_path, img_cv2)
            logger.info(f"Saved result to {output_path}")

        return img_cv2
