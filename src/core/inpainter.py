import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class Inpainter:
    def __init__(self):
        pass

    def clean_bubble(self, image: np.ndarray, bubble_bbox: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cleans the text bubble (whitens it) using Otsu thresholding within the bounding box.

        Args:
            image: Full original image (BGR).
            bubble_bbox: [x1, y1, x2, y2]

        Returns:
            Tuple(cleaned_image_patch, bubble_mask_contour)
            - cleaned_image_patch: The roi with the bubble interior whitened.
            - bubble_mask_contour: The specific contour of the bubble found.
        """
        x1, y1, x2, y2 = bubble_bbox
        roi = image[y1:y2, x1:x2].copy()

        if roi.size == 0:
            return roi, np.array([])

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Otsu thresholding to find the bubble area (assuming bubble is lighter than text/art)
        # We invert it because we want the white background to be the mask we target?
        # Usually text is black, bubble is white.
        # Threshold: White pixels become 255 (foreground), others 0.
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return roi, np.array([])

        # Assume the largest contour is the bubble shape
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the bubble interior
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)

        # Inpaint: Set the masked area to white (255, 255, 255)
        # This erases the original text and art inside the bubble
        roi[mask == 255] = (255, 255, 255)

        return roi, largest_contour

    def render_text(self, roi: np.ndarray, text: str, font_path: str, contour: np.ndarray) -> np.ndarray:
        """
        Renders text inside the cleaned bubble ROI.
        """
        if contour.size == 0:
            return roi

        # Calculate bounding rect of the actual bubble shape inside the ROI
        cx, cy, cw, ch = cv2.boundingRect(contour)

        # Convert to PIL for text drawing
        pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Dynamic font sizing and wrapping
        # Start large and shrink until it fits
        target_width_ratio = 0.8
        target_height_ratio = 0.8

        font_size = 40 # Start large
        min_font_size = 10

        # Try to fit text
        while font_size >= min_font_size:
            try:
                font = ImageFont.truetype(font_path, size=font_size)
            except IOError:
                # Fallback to default if font load fails
                font = ImageFont.load_default()
                logger.warning(f"Could not load font {font_path}, using default.")

            # Wrap text based on width
            # Estimate char width (approximate)
            avg_char_width = font_size * 0.5
            max_chars_per_line = int((cw * target_width_ratio) / avg_char_width)
            if max_chars_per_line < 1: max_chars_per_line = 1

            wrapped_text = textwrap.fill(text, width=max_chars_per_line, break_long_words=True)

            # Check height
            bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            if text_width <= cw * 0.9 and text_height <= ch * target_height_ratio:
                break # Fits!

            font_size -= 2

        # Final render
        # Center text in the bubble bounding rect
        final_x = cx + (cw - text_width) // 2
        final_y = cy + (ch - text_height) // 2

        draw.multiline_text((final_x, final_y), wrapped_text, font=font, fill=(0, 0, 0), align="center")

        # Convert back to OpenCV
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
