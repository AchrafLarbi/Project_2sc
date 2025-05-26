import os
import google.generativeai as genai
import google
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import os
import base64
import google.generativeai as genai

class GeminiOCR:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def _read_image_base64(self, image_path: str):
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
        return {
            "mime_type": "image/jpeg",  # or image/png
            "data": image_bytes,
        }

    def extract_text_from_image(self, image_path: str, prompt: str = None):
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return None

        # Check if this is a plate image or vehicle image based on the filename
        is_plate_image = "plate_" in image_path.lower()
        
        if is_plate_image:
            prompt = """Extract the license plate text from the image. If the plate is blurred or unreadable, respond with 'NO_PLATE_DETECTED'. Respond only with the plate number in the format XXX NNNN (e.g., 7AC 3391) or 'NO_PLATE_DETECTED', and nothing else."""
        else:
            prompt = """This is a vehicle image. Look for  license plate on the vehicle. Extract the license plate text """

        try:
            image_data = self._read_image_base64(image_path)
            response = self.model.generate_content([
                image_data,
                prompt
            ])

            result = response.text.strip()
            return None if result == 'NO_PLATE_DETECTED' else result

        except Exception as e:
            print(f"Error during OCR: {e}")
            return None