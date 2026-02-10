import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)

class SileroVAD:
    """Wrapper for Silero VAD."""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        """Initialize the VAD model."""
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.model = None
        self.utils = None
        self._init_model()

    def _init_model(self):
        try:
            # Load Silero VAD from torch hub
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            logger.info("Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if the audio chunk contains speech.

        Args:
            audio_chunk: Float32 numpy array of audio samples

        """
        if self.model is None:
            return False

        # Convert to tensor
        tensor = torch.from_numpy(audio_chunk)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            speech_prob = self.model(tensor, self.sample_rate).item()

        return speech_prob > self.threshold
