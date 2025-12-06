
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.utils.logger import setup_logger

logger = setup_logger("ImageProcessor")

class ImageProcessor:
    @staticmethod
    def sort_bubbles(bubbles, row_threshold=50):
        """
        Sort bubbles: Right-to-Left, Top-to-Bottom (Manga style).
        bubbles: List of [x1, y1, x2, y2, conf, cls]
        """
        if not bubbles:
            return []

        # Calculate centers
        bubble_data = []
        for b in bubbles:
            x1, y1, x2, y2 = map(int, b[:4])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            bubble_data.append({
                'box': b,
                'cx': center_x,
                'cy': center_y
            })

        # Sort by Y first
        bubble_data.sort(key=lambda x: x['cy'])

        rows = []
        if not bubble_data:
            return []

        current_row = [bubble_data[0]]

        for bubble in bubble_data[1:]:
            if abs(bubble['cy'] - current_row[0]['cy']) < row_threshold:
                current_row.append(bubble)
            else:
                rows.append(current_row)
                current_row = [bubble]
        rows.append(current_row)

        # Sort rows by X descending (Right to Left)
        sorted_bubbles = []
        for row in rows:
            row.sort(key=lambda x: -x['cx'])
            sorted_bubbles.extend([b['box'] for b in row])

        return sorted_bubbles

    @staticmethod
    def clean_bubble(image_roi):
        """
        Clean text from the bubble ROI (inpainting).
        Returns: cleaned_roi, mask
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)

        # Threshold to find dark text on light background
        # Or adaptive thresholding
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Dilate mask slightly to cover artifacts
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Inpaint
        cleaned = cv2.inpaint(image_roi, mask, 3, cv2.INPAINT_TELEA)

        return cleaned, mask

    @staticmethod
    def add_text(image, text, font_path, box):
        """
        Add text to the image within the box.
        """
        x1, y1, x2, y2 = map(int, box[:4])
        width = x2 - x1
        height = y2 - y1

        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # Dynamic font size
        font_size = 20
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        # Wrap text and adjust font size to fit
        # Simplified version: just basic wrapping

        # ... logic for text wrapping ...
        # For now, let's just draw it centered

        # Calculate text size
        # bbox = draw.textbbox((0, 0), text, font=font)
        # text_w = bbox[2] - bbox[0]
        # text_h = bbox[3] - bbox[1]

        # Very basic fitting loop
        for s in range(40, 8, -2):
            try:
                font = ImageFont.truetype(font_path, s)
            except:
                font = ImageFont.load_default()
                break

            # Estimate width (approximate)
            # Better implementation would use textwrap
            if len(text) * (s/2) < width * (height/s) * 1.5: # Heuristic
                font_size = s
                break

        # Simple word wrap
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            # check width
            line_str = " ".join(current_line)
            bbox = draw.textbbox((0, 0), line_str, font=font)
            if (bbox[2] - bbox[0]) > width * 0.9:
                current_line.pop()
                lines.append(" ".join(current_line))
                current_line = [word]
        lines.append(" ".join(current_line))

        # Draw text
        text_h_total = len(lines) * font_size * 1.2
        start_y = y1 + (height - text_h_total) // 2

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            line_x = x1 + (width - line_w) // 2
            line_y = start_y + i * font_size * 1.2

            # Draw outline
            outline_color = "white"
            text_color = "black"
            thickness = 2

            draw.text((line_x-thickness, line_y), line, font=font, fill=outline_color)
            draw.text((line_x+thickness, line_y), line, font=font, fill=outline_color)
            draw.text((line_x, line_y-thickness), line, font=font, fill=outline_color)
            draw.text((line_x, line_y+thickness), line, font=font, fill=outline_color)

            draw.text((line_x, line_y), line, font=font, fill=text_color)

        return np.array(img_pil)
