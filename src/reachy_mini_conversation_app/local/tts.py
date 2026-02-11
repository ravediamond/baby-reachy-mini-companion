import asyncio
import logging
from typing import Tuple

import numpy as np
from kokoro_onnx import Kokoro


logger = logging.getLogger(__name__)

class LocalTTS:
    """Wrapper for Kokoro-ONNX TTS."""

    def __init__(self, model_path: str = "kokoro-v0_19.onnx", voices_path: str = "voices.npz"):
        """Initialize the TTS engine."""
        self.model_path = model_path
        self.voices_path = voices_path
        self.kokoro: Kokoro | None = None
        self._init_model()

    def _init_model(self):
        try:
            import os

            from huggingface_hub import hf_hub_download

            # v0.19 files were removed from main, pin to last commit with them
            revision = "e9d173129d407bf1378c402aba163de4dde2615e"

            if not os.path.exists(self.model_path):
                logger.info(f"Downloading Kokoro model to {self.model_path}...")
                self.model_path = hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M",
                    filename="kokoro-v0_19.onnx",
                    local_dir=".",
                    revision=revision,
                )

            if not os.path.exists(self.voices_path):
                logger.info(f"Downloading Kokoro voices to {self.voices_path}...")
                self._download_and_convert_voices()

            self.kokoro = Kokoro(self.model_path, self.voices_path)
            logger.info(f"Kokoro TTS loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load Kokoro TTS: {e}")
            logger.warning("Ensure kokoro-v0_19.onnx and voices.npz are present!")

    def _download_and_convert_voices(self) -> None:
        """Download voices-v1.0.bin and convert to voices.npz."""
        import os
        from urllib.request import urlretrieve

        import torch

        bin_path = "voices-v1.0.bin"
        url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

        try:
            if not os.path.exists(bin_path):
                logger.info(f"Downloading {url} ...")
                urlretrieve(url, bin_path)

            data = torch.load(bin_path, weights_only=False, map_location="cpu")
            np_data = {k: v.numpy() if hasattr(v, "numpy") else np.array(v) for k, v in data.items()}
            np.savez(self.voices_path, **np_data)
            logger.info(f"Converted {len(np_data)} voices to {self.voices_path}")

            os.remove(bin_path)
        except Exception as e:
            logger.error(f"Failed to download/convert voices: {e}")
            raise

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
