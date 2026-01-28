import logging
import asyncio
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from typing import Tuple

logger = logging.getLogger(__name__)

class LocalTTS:
    """Wrapper for Kokoro-ONNX TTS."""

    def __init__(self, model_path: str = "kokoro-v0_19.onnx", voices_path: str = "voices.json"):
        self.model_path = model_path
        self.voices_path = voices_path
        self.kokoro = None
        self._init_model()

    def _init_model(self):
        try:
            import os
            from huggingface_hub import hf_hub_download

            if not os.path.exists(self.model_path):
                logger.info(f"Downloading Kokoro model to {self.model_path}...")
                self.model_path = hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M",
                    filename="kokoro-v0_19.onnx",
                    local_dir=".",
                )
            
            if not os.path.exists(self.voices_path):
                logger.info(f"Downloading Kokoro voices to {self.voices_path}...")
                self.voices_path = hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M",
                    filename="voices.json",
                    local_dir=".",
                )

            self.kokoro = Kokoro(self.model_path, self.voices_path)
            logger.info(f"Kokoro TTS loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load Kokoro TTS: {e}")
            logger.warning("Ensure kokoro-v0_19.onnx and voices.json are present!")

    async def synthesize(self, text: str, voice: str = "af_sarah", speed: float = 1.0) -> Tuple[int, np.ndarray]:
        """Synthesize text to audio. Returns (sample_rate, audio_float32)."""
        if self.kokoro is None:
            return 24000, np.zeros(0, dtype=np.float32)

        try:
            # Run in thread pool to avoid blocking async loop
            samples, sample_rate = await asyncio.to_thread(
                self.kokoro.create,
                text,
                voice=voice,
                speed=speed,
                lang="en-us"
            )
            return sample_rate, samples
        except Exception as e:
            logger.error(f"TTS Synthesis failed: {e}")
            return 24000, np.zeros(0, dtype=np.float32)
