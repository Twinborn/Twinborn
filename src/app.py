
import os
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from src.utils import config
from src.utils.logger import setup_logger
from src.core.detector import BubbleDetector
from src.core.ocr import OCRProcessor
from src.core.translator import Translator
from src.core.image_processing import ImageProcessor

logger = setup_logger("App")

class MangaProcessor:
    def __init__(self):
        logger.info("Initializing MangaProcessor...")
        self.detector = BubbleDetector()
        self.ocr = OCRProcessor()
        self.translator = Translator()
        logger.info("MangaProcessor Initialized.")

    def process(self, image, target_language, font_choice, progress=None):
        logger.info(f"Processing image for language: {target_language}")

        # 1. Detect Bubbles
        if progress: progress(0.1, desc="Detecting bubbles...")
        detections = self.detector.detect(image)

        # Filter for bubbles (Class 0)
        bubbles = [
            d for d in detections
            if int(d[5]) == config.BUBBLE_CLASS_ID
            and float(d[4]) >= config.CONFIDENCE_THRESHOLD
        ]

        if not bubbles:
            logger.warning("No bubbles detected.")
            return image

        # 2. Sort Bubbles
        bubbles = ImageProcessor.sort_bubbles(bubbles, config.ROW_THRESHOLD)

        # 3. Process each bubble
        img_array = np.array(image)
        # Work on a copy
        final_image = img_array.copy()

        total_bubbles = len(bubbles)
        for i, bubble in enumerate(bubbles):
            if progress:
                progress(0.2 + 0.7 * (i / total_bubbles), desc=f"Processing bubble {i+1}/{total_bubbles}...")

            x1, y1, x2, y2 = map(int, bubble[:4])

            # Clamp coordinates
            h, w = img_array.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            bubble_roi = img_array[y1:y2, x1:x2]
            if bubble_roi.size == 0: continue

            # OCR
            text = self.ocr.extract_text(bubble_roi)
            if not text or len(text) < config.MIN_TEXT_LENGTH:
                continue

            logger.info(f"Bubble {i+1} Text: {text}")

            # Translate
            translated_text = self.translator.translate(text, target_language)
            logger.info(f"Bubble {i+1} Trans: {translated_text}")

            # Inpaint (Clean)
            cleaned_roi, _ = ImageProcessor.clean_bubble(bubble_roi)

            # Replace the ROI in the final image with cleaned version temporarily
            # (We need to put the cleaned ROI back into the main image context
            # OR we can draw text on the main image. Drawing on main image is better for handling overlaps
            # but simpler to draw on ROI and paste back)

            # Let's paste cleaned ROI back first
            final_image[y1:y2, x1:x2] = cleaned_roi

            # Add Text
            # We call add_text on the whole image (or a larger crop) to handle overflows better?
            # Or just update the final_image region

            # For simplicity, use the helper which takes an image, text, font, and box.
            # But the helper was designed to return a modified image.

            # Let's refine the approach:
            # 1. Clean the ROI.
            # 2. Put cleaned ROI into final_image.
            # 3. Draw text on final_image within the bounding box.

            final_image = ImageProcessor.add_text(
                final_image,
                translated_text,
                font_choice,
                [x1, y1, x2, y2]
            )

        return Image.fromarray(final_image)

# Global instance
processor = None

def get_processor():
    global processor
    if processor is None:
        processor = MangaProcessor()
    return processor

def predict(img, target_language, font_choice, progress=gr.Progress()):
    try:
        proc = get_processor()
        result = proc.process(img, target_language, font_choice, progress)
        return result
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        raise gr.Error(f"Processing failed: {str(e)}")

def create_interface():
    with gr.Blocks(title=config.TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {config.TITLE}")
        gr.Markdown(config.DESCRIPTION)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Manga Page")
                target_lang = gr.Dropdown(
                    choices=config.LANGUAGES,
                    value=config.DEFAULT_LANGUAGE,
                    label="Target Language"
                )
                font_select = gr.Dropdown(
                    choices=config.FONTS,
                    value=config.DEFAULT_FONT,
                    label="Font"
                )
                submit_btn = gr.Button("Translate", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Translated Page")

        submit_btn.click(
            fn=predict,
            inputs=[input_image, target_lang, font_select],
            outputs=output_image
        )

    return demo

if __name__ == "__main__":
    # Ensure fonts exist
    if not os.path.exists("fonts"):
        os.makedirs("fonts")

    demo = create_interface()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
