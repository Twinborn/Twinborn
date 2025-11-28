import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from src.core import MangaTranslator
import tempfile

st.set_page_config(page_title="Manga Translator AI", layout="wide")

st.title("Manga Translator AI 🚀")
st.markdown("Upload a manga page, and we will translate the bubbles for you!")

# Sidebar Configuration
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
model_path = st.sidebar.text_input("YOLO Model Path", value="best.pt")
font_path = st.sidebar.text_input("Font Path", value="font.ttf")
ocr_scale = st.sidebar.slider("OCR Scale", 1.0, 3.0, 1.2, 0.1)
target_lang = st.sidebar.text_input("Target Language", value="Turkish")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tfile.write(uploaded_file.read())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(tfile.name)

    if st.button("Translate"):
        if not api_key:
            st.error("Please provide a Gemini API Key.")
        elif not os.path.exists(model_path):
             st.error(f"Model not found at {model_path}")
        else:
            with st.spinner("Translating... This might take a while."):
                try:
                    translator = MangaTranslator(
                        model_path=model_path,
                        gemini_api_key=api_key,
                        font_path=font_path,
                        ocr_scale=ocr_scale,
                        target_lang=target_lang
                    )

                    result_img = translator.translate_image(tfile.name)

                    if result_img is not None:
                        # Convert BGR to RGB for Streamlit
                        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                        with col2:
                            st.subheader("Translated Image")
                            st.image(result_rgb)

                        # Download button
                        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                        is_success, buffer = cv2.imencode(".jpg", result_bgr)
                        if is_success:
                            st.download_button(
                                label="Download Translated Image",
                                data=buffer.tobytes(),
                                file_name="translated_manga.jpg",
                                mime="image/jpeg"
                            )
                    else:
                        st.error("Translation failed or returned empty result.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.exception(e)

st.markdown("---")
st.markdown("Powered by YOLO, EasyOCR, and Gemini.")
