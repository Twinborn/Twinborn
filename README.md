# Manga Translator Pro 🎌

Welcome to **Manga Translator Pro**! This tool allows you to automatically translate manga pages from any language to your preferred language using the power of AI.

It uses **YOLO AI** to find speech bubbles and **Google Gemini AI** to translate them naturally.

## 🚀 How to Install and Run

### For Windows Users 🪟

1. **Download** the project zip file and extract it to a folder.
2. Inside the folder, find the file named `install_and_run.bat`.
3. **Double-click** it.
4. A black window (terminal) will open. It will automatically download all necessary "brains" (libraries) for the AI. This might take a few minutes the first time.
5. Once finished, it will give you a link (usually `http://127.0.0.1:7860`). Copy and paste that into your web browser, or it might just say "Running".
6. Use the App in your browser!

### For Mac / Linux Users 🍎🐧

1. Open your terminal in the project folder.
2. Run `chmod +x install_and_run.sh`.
3. Run `./install_and_run.sh`.

---

## 🎮 How to Use the App

1. **Get an API Key**:
   - You need a "password" for the translator brain (Google Gemini).
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
   - Click "Create API Key".
   - Copy the key (it starts with `AIza...`).

2. **Open the App**:
   - Paste your **API Key** into the box labeled "Google Gemini API Key".

3. **Upload Manga**:
   - Drag and drop your manga page image into the "Original Page" box.

4. **Choose Settings**:
   - Select your **Target Language** (e.g., English).
   - Select a **Font** style.

5. **Click Translate!**:
   - Hit the **Translate 🚀** button and wait a few seconds.
   - The translated page will appear on the right!

---

## 🌟 Features

- **Smart Detection**: Uses a custom AI model to find bubbles and text specifically.
- **Multilingual Support**: Supports translating *from* almost anything (thanks to Surya OCR) *to* many languages.
- **Auto-Cleaning**: Automatically whitens the bubbles before writing new text.
- **Context Aware**: Uses Gemini AI to understand the context of the sentence for better translations.

## 🔮 Future Plans

- **Browser Extension**: In the future, we plan to bring this technology directly to your browser, so you can translate manga reading websites in real-time!

## 🛠 Troubleshooting

- **"It's slow!"**: If you don't have a powerful graphics card (GPU), the AI runs on your CPU, which takes longer. Be patient!
- **"Translation Failed"**: Check if your API Key is correct and has credits (Google offers a free tier).

---
*Created for you with ❤️*
