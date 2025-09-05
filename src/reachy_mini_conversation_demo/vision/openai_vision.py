import base64
import logging
import os
import cv2
from openai import OpenAI
from .processors import VisionConfig

logger = logging.getLogger(__name__)


class OpenAIVisionProcessor:
    def __init__(self, config: VisionConfig = None):
        self.config = config or VisionConfig()
        self._initialized = False
        self.client = None

    def initialize(self):
        """Initialize OpenAI client with proper error handling"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not found in environment variables")
                return False

            self.client = OpenAI(api_key=api_key)

            # Smoke test the API/key
            try:
                _ = self.client.models.list()
                self._initialized = True
                logger.info("OpenAI Vision processor initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to connect to OpenAI API: {e}")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI Vision processor: {e}")
            return False

    def process_image(
        self, cv2_image, prompt="Briefly describe what you see in one sentence."
    ):
        """Process image using OpenAI (Responses API) with retry logic"""
        if not self._initialized:
            return "OpenAI Vision processor not initialized"

        for attempt in range(self.config.max_retries):
            try:
                # Convert image to base64
                rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                ok, jpeg_buffer = cv2.imencode(
                    ".jpg",
                    rgb_image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
                )
                if not ok:
                    return "Failed to encode image"
                image_base64 = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")

                # Responses API with input_image
                response = self.client.responses.create(
                    model=self.config.openai_model,  # e.g., gpt-4.1 or gpt-4.1-mini
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                                },
                            ],
                        }
                    ],
                    max_output_tokens=300,
                )

                # Unified text accessor
                text = (response.output_text or "").strip()
                return text if text else "No response"

            except Exception as e:
                logger.error(f"OpenAI Vision API error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    import time

                    time.sleep(self.config.retry_delay)
                else:
                    return f"OpenAI Vision processing failed after {self.config.max_retries} attempts"

    def get_model_info(self):
        return {
            "processor_type": "openai",
            "initialized": self._initialized,
            "model": self.config.openai_model,
        }
