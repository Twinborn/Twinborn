
# Manga Translator Pro

Welcome to the Manga Translator Pro! This application allows you to automatically translate manga pages using AI. It detects speech bubbles, reads the text using advanced OCR, translates it using Google Gemini, and replaces the text in the image.

## Features
- **Smart Detection**: Uses YOLOv8 to find speech bubbles.
- **High-Accuracy OCR**: Uses Surya OCR for precise text recognition.
- **Context-Aware Translation**: Uses Google Gemini to translate manga text naturally.
- **Easy Interface**: Simple web interface to upload and translate.

## Prerequisites

Before you start, you need to have **Python 3.10** or higher installed on your computer.

### Step 1: Install Python
If you don't have Python, download it from [python.org](https://www.python.org/downloads/).

### Step 2: Get Your Google API Key
To use the translation feature, you need a free Google API Key.
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Create an API key.
3. Save it for later.

## Installation (For Beginners)

1. **Download the Zip file** provided and extract it to a folder.
2. **Open the folder** where you extracted the files.

### Windows
1. Double-click on `install_and_run.bat`.
2. It will ask for your Google API Key. Paste it and press Enter.
3. The app will install everything and open in your browser.

### Mac / Linux
1. Open Terminal.
2. Navigate to the folder: `cd path/to/folder`
3. Run: `bash install_and_run.sh`

## How to Use
1. Once the app starts, you will see a web page.
2. **Upload** an image of a manga page.
3. Select your **Target Language** (e.g., English, Turkish).
4. Click **Translate**.
5. Wait a few seconds, and your translated page will appear!

## Troubleshooting
- **Installation Stuck?** Be patient, installing AI models can take a few minutes.
- **Error loading models?** Make sure you have a stable internet connection for the first run.
- **Translation failed?** Check if your API Key is correct.

## Note on Performance
Surya OCR is very accurate but can be resource-intensive. If you have a dedicated GPU (NVIDIA), make sure to install PyTorch with CUDA support for faster processing.
