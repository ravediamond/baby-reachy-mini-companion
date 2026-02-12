import time
import base64
import logging
import threading
from typing import Any, Dict
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


@dataclass
class VisionConfig:
    """Configuration for vision processing."""

    vision_interval: float = 5.0
    max_new_tokens: int = 64
    jpeg_quality: int = 85
    max_retries: int = 3
    retry_delay: float = 1.0
    continuous_mode: bool = False


class VisionProcessor:
    """Handles vision via Remote/Local API (Ollama/vLLM) only."""

    def __init__(self, vision_config: VisionConfig | None = None):
        """Initialize the vision processor."""
        self.vision_config = vision_config or VisionConfig()
        self._initialized = False

    def initialize(self) -> bool:
        """Mark as initialized (API mode needs no loading)."""
        logger.info(f"Vision Processor initialized in API-only mode (URL={config.LOCAL_LLM_URL})")
        self._initialized = True
        return True

    def _process_api(self, cv2_image: NDArray[np.uint8], prompt: str) -> str:
        """Process image using OpenAI-compatible API."""
        try:
            from openai import OpenAI

            client = OpenAI(base_url=config.LOCAL_LLM_URL, api_key=config.LOCAL_LLM_API_KEY)

            # Use the same model as the chat LLM
            model_name = config.LOCAL_LLM_MODEL or "ministral-3:3b"

            logger.debug(f"Encoding image for VLM (model={model_name} at {config.LOCAL_LLM_URL})...")

            # Encode image
            success, jpeg_buffer = cv2.imencode(
                ".jpg",
                cv2_image,
                [cv2.IMWRITE_JPEG_QUALITY, self.vision_config.jpeg_quality],
            )
            if not success:
                logger.error("Failed to encode image for VLM")
                return "Failed to encode image"

            image_base64 = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")

            logger.debug("Sending vision request...")
            start_time = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                max_tokens=self.vision_config.max_new_tokens,
            )
            duration = time.time() - start_time
            logger.debug(f"Vision request took {duration:.2f}s")

            content = response.choices[0].message.content
            return content if content else "No response from VLM"

        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return f"Vision Error: {e}"

    def process_image(
        self,
        cv2_image: NDArray[np.uint8],
        prompt: str = "Briefly describe what you see in one sentence.",
    ) -> str:
        """Process CV2 image and return description."""
        if not self._initialized:
            return "Vision model not initialized"

        return self._process_api(cv2_image, prompt)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "initialized": self._initialized,
            "mode": "api",
            "url": config.LOCAL_LLM_URL,
            "model": config.LOCAL_LLM_MODEL,
        }


class VisionManager:
    """Manages periodic vision processing (if enabled) and API access."""

    def __init__(self, camera: Any, vision_config: VisionConfig | None = None):
        """Initialize vision manager with camera and configuration."""
        self.camera = camera
        self.vision_config = vision_config or VisionConfig()
        self.vision_interval = self.vision_config.vision_interval
        self.processor = VisionProcessor(self.vision_config)

        self._last_processed_time = 0.0
        self.latest_description: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Initialize processor
        if not self.processor.initialize():
            logger.error("Failed to initialize vision processor")
            raise RuntimeError("Vision processor initialization failed")

    def start(self) -> None:
        """Start the vision processing loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._working_loop, daemon=True)
        self._thread.start()
        logger.info("Local vision processing started")

    def stop(self) -> None:
        """Stop the vision processing loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.info("Local vision processing stopped")

    def _working_loop(self) -> None:
        """Vision processing loop (runs in separate thread)."""
        if not self.vision_config.continuous_mode:
            logger.info("Vision manager running in on-demand mode (continuous processing disabled)")
            return

        logger.debug(f"Vision loop started with interval {self.vision_interval}s")
        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                if current_time - self._last_processed_time >= self.vision_interval:
                    frame = self.camera.get_latest_frame()
                    if frame is not None:
                        description = self.processor.process_image(
                            frame,
                            "Briefly describe what you see in one sentence.",
                        )
                        self._last_processed_time = current_time

                        if description and not description.startswith(("Vision", "Failed", "Error")):
                            self.latest_description = description
                            logger.info(f"Vision update: {description}")
                    else:
                        pass

                time.sleep(1.0)

            except Exception:
                logger.exception("Vision processing loop error")
                time.sleep(5.0)

        logger.info("Vision loop finished")


def initialize_vision_manager(camera_worker: Any, continuous_mode: bool = False) -> VisionManager | None:
    """Initialize vision manager in API-only mode."""
    try:
        # Configure vision processing
        vision_config = VisionConfig(
            vision_interval=5.0,
            max_new_tokens=64,
            jpeg_quality=85,
            max_retries=3,
            retry_delay=1.0,
            continuous_mode=continuous_mode,
        )

        # Initialize vision manager
        vision_manager = VisionManager(camera_worker, vision_config)
        return vision_manager

    except Exception as e:
        logger.error(f"Failed to initialize vision manager: {e}")
        return None
