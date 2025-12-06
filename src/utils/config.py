
import os

# Configuration constants
TITLE = "Manga Translator Pro"
DESCRIPTION = """
**AI-Powered Manga Translation System**
- YOLOv8 for Bubble Detection
- Surya OCR for Text Recognition
- Gemini AI for Context-Aware Translation
"""

# Default Settings
DEFAULT_LANGUAGE = "Turkish"
DEFAULT_FONT = "fonts/animeace_i.ttf"
CONFIDENCE_THRESHOLD = 0.5
BUBBLE_CLASS_ID = 0
TEXT_CLASS_ID = 2

# Row sorting threshold (pixels)
ROW_THRESHOLD = 50

# Minimum text length to translate
MIN_TEXT_LENGTH = 2

# Supported Languages
LANGUAGES = [
    "Turkish", "English", "Japanese", "Portuguese",
    "Spanish", "French", "German", "Italian", "Russian"
]

# Supported Fonts
FONTS = [
    "fonts/animeace_i.ttf",
    "fonts/mangati.ttf"
]
