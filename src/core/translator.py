import google.generativeai as genai
import logging
import time

logger = logging.getLogger(__name__)

class Translator:
    def __init__(self, api_key: str = None):
        """
        Initializes the Translator.

        Args:
            api_key: Google Gemini API Key. Can be passed later if not available on init.
        """
        self.model = None
        self.api_key = api_key
        if api_key:
            self.configure(api_key)

    def configure(self, api_key: str):
        """Configures the Gemini API with a new key."""
        try:
            self.api_key = api_key
            genai.configure(api_key=self.api_key)
            # Using 'gemini-1.5-flash' as it's faster and cheaper, or fallback to pro.
            # User's code had 'gemini-flash-latest'.
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini API configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            self.model = None

    def translate(self, text: str, target_lang: str = "English") -> str:
        """
        Translates text using Gemini.
        """
        if not text or not text.strip():
            return ""

        if not self.model:
            logger.warning("Translator model not configured. Returning original text.")
            return "Error: API Key missing"

        prompt = f"""
        You are a professional manga translator.
        The text below is extracted from a manga page and might have OCR errors.

        Task:
        1. Correct any OCR typos or broken words in the source text intuitively.
        2. Translate the corrected text to {target_lang} naturally, fitting for a comic book context.
        3. Output ONLY the translation. Do not include notes or the original text.

        Original Text: {text}
        """

        try:
            # Simple retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    if response.text:
                        return response.text.strip()
                except Exception as e:
                    if "429" in str(e): # Rate limit
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        raise e
            return text # Return original if failed

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text
