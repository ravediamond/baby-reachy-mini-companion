import logging

import numpy as np
from faster_whisper import WhisperModel


logger = logging.getLogger(__name__)

class LocalSTT:
    """Wrapper for Faster-Whisper."""

    def __init__(self, model_size: str = "base.en", device: str = "auto", compute_type: str = "int8"):
        """Initialize the STT engine."""
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            logger.info(f"Loading Faster-Whisper model: {self.model_size}")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("Faster-Whisper loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper: {e}")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio array to text.

        Args:
            audio: Float32 numpy array (16kHz mono)

        """
        if self.model is None:
            return ""

        try:
            segments, _ = self.model.transcribe(audio, beam_size=5)
            text = " ".join([segment.text for segment in segments]).strip()
            return text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
