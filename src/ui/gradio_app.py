import gradio as gr
import os
import logging
from PIL import Image
import numpy as np
from typing import List

# Import our core modules
from src.core.detector import Detector
from src.core.ocr import OCRProcessor
from src.core.translator import Translator
from src.core.inpainter import Inpainter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "models/model.pt"
FONTS_DIR = "assets/fonts/"

class MangaApp:
    def __init__(self):
        self.detector = Detector(MODEL_PATH)
        self.ocr = OCRProcessor() # Singleton
        self.translator = Translator() # Key added later
        self.inpainter = Inpainter()

    def process_manga(self, image, api_key, target_lang, font_name, progress=gr.Progress()):
        """
        Main pipeline function.
        """
        if not api_key:
            raise gr.Error("Please enter your Google Gemini API Key.")

        # Configure Translator
        self.translator.configure(api_key)

        if image is None:
            return None

        progress(0.1, desc="Detecting Bubbles...")
        logger.info("Starting detection...")

        # 1. Detection
        bubbles = self.detector.detect(image)
        logger.info(f"Detected {len(bubbles)} bubbles.")

        if not bubbles:
            return image

        # Convert to numpy for OpenCV processing
        final_image = np.array(image)

        # Filter bubbles (optional: sort them)
        # Simple sorting: Top-to-bottom, Right-to-left (standard manga)
        bubbles.sort(key=lambda b: (b['bbox'][1] // 50, -b['bbox'][0]))

        font_path = os.path.join(FONTS_DIR, font_name)

        total = len(bubbles)
        for i, bubble in enumerate(bubbles):
            progress(0.1 + (0.8 * (i / total)), desc=f"Processing Bubble {i+1}/{total}")

            bbox = bubble['bbox']
            x1, y1, x2, y2 = bbox

            # 2. OCR
            # Use detector's text crops if available to guide OCR
            original_text = self.ocr.process_bubble(final_image, bubble)

            if not original_text or len(original_text) < 2:
                continue # Skip empty bubbles

            logger.info(f"Bubble {i+1} OCR: {original_text}")

            # 3. Translate
            translated_text = self.translator.translate(original_text, target_lang)
            logger.info(f"Bubble {i+1} TR: {translated_text}")

            # 4. Inpaint & Render
            # We work on the ROI
            try:
                roi, contour = self.inpainter.clean_bubble(final_image, bbox)
                roi_with_text = self.inpainter.render_text(roi, translated_text, font_path, contour)

                # Paste back
                final_image[y1:y2, x1:x2] = roi_with_text
            except Exception as e:
                logger.error(f"Error rendering bubble {i+1}: {e}")
                continue

        progress(1.0, desc="Done!")
        return Image.fromarray(final_image)

# --- GRADIO INTERFACE CONSTRUCTION ---
def create_ui():
    app_instance = MangaApp()

    # Workaround for Gradio 4.x / 5.x / 6.x compatibility issues with theme kwarg
    # Initialize Blocks first, then set theme property if possible, or just default.
    # The error "TypeError: BlockContext.__init__() got an unexpected keyword argument 'theme'"
    # implies that in this specific version, 'theme' is not an init arg for BlockContext (parent of Blocks).
    # However, documentation says it is. It might be a library version quirk in this environment.
    # We will try initializing without theme and setting it manually or ignoring it to ensure stability.

    try:
        demo = gr.Blocks(title="Manga Translator Pro", theme=gr.themes.Soft())
    except TypeError:
        logger.warning("Could not set theme via constructor, falling back to default.")
        demo = gr.Blocks(title="Manga Translator Pro")
        try:
             demo.theme = gr.themes.Soft()
        except:
             pass

    with demo:
        gr.Markdown("# 🎌 Manga Translator Pro")
        gr.Markdown("Upload a manga page, provide your Gemini API Key, and translate it automatically!")

        with gr.Row():
            with gr.Column(scale=1):
                # Inputs
                api_key_input = gr.Textbox(
                    label="Google Gemini API Key",
                    placeholder="AIzaSy...",
                    type="password",
                    info="Get your key from Google AI Studio"
                )

                input_image = gr.Image(type="pil", label="Original Page")

                target_lang = gr.Dropdown(
                    choices=["English", "Spanish", "French", "German", "Turkish", "Portuguese", "Russian", "Japanese"],
                    value="English",
                    label="Target Language"
                )

                # Scan fonts dir
                font_files = [f for f in os.listdir(FONTS_DIR) if f.endswith(".ttf")] if os.path.exists(FONTS_DIR) else ["arial.ttf"]
                font_selector = gr.Dropdown(
                    choices=font_files,
                    value=font_files[0] if font_files else None,
                    label="Font"
                )

                btn_run = gr.Button("Translate 🚀", variant="primary")

            with gr.Column(scale=2):
                output_image = gr.Image(type="pil", label="Translated Page")

        # Event Logic
        btn_run.click(
            fn=app_instance.process_manga,
            inputs=[input_image, api_key_input, target_lang, font_selector],
            outputs=[output_image]
        )

        # Example loading
        # Check if examples exist
        examples_dir = "assets/examples"
        if os.path.exists(examples_dir) and os.listdir(examples_dir):
            example_files = [os.path.join(examples_dir, f) for f in os.listdir(examples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if example_files:
                 gr.Examples(
                    examples=example_files,
                    inputs=input_image
                )

    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=True)
