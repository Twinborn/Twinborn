
import os
import google.generativeai as genai
from src.utils.logger import setup_logger

logger = setup_logger("Translator")

class Translator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("Google API Key not found. Translation might fail.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.history = []

    def translate(self, text, target_lang="Turkish"):
        if not text:
            return ""

        try:
            prompt = f"""
            Translate the following manga text to {target_lang}.
            Keep the tone and context appropriate for manga/comics.
            If the text is a sound effect (SFX), try to translate it as an SFX or keep it if it fits.
            Only output the translation, no explanations.

            Text: "{text}"
            """

            # Simple stateless translation for robustness,
            # though we could use chat session for context if needed.
            response = self.model.generate_content(prompt)
            translated_text = response.text.strip()

            # Remove quotes if the model adds them
            if translated_text.startswith('"') and translated_text.endswith('"'):
                translated_text = translated_text[1:-1]

            return translated_text
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text  # Return original text on failure
