import csv
import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioClassifier:
    """YAMNet-based audio event classifier."""

    def __init__(self, model_path: str = "cache/models--yamnet/yamnet.onnx", map_path: str = "cache/models--yamnet/yamnet_class_map.csv"):
        self.model_path = model_path
        self.map_path = map_path
        self.session = None
        self.class_names = []
        self._load_model()

    def _load_model(self):
        try:
            # Load Class Map
            if not Path(self.map_path).exists():
                logger.error(f"Class map not found: {self.map_path}")
                return

            with open(self.map_path, 'r') as f:
                reader = csv.reader(f)
                next(reader) # Skip header
                self.class_names = [row[2] for row in reader]

            # Load ONNX Model
            if not Path(self.model_path).exists():
                logger.error(f"Model not found: {self.model_path}")
                return

            self.session = ort.InferenceSession(self.model_path)
            logger.info("YAMNet Audio Classifier loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Audio Classifier: {e}")

    def classify(self, audio: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Classify audio chunk.
        
        Args:
            audio: 16kHz float32 mono audio (should be ~0.975s for YAMNet, typically 15600 samples)
                   YAMNet expects input shape [N] where N is waveform length.
        """
        if self.session is None:
            return []

        try:
            # YAMNet typically expects chunks of 0.975s (15600 samples). 
            # If longer, we take the center or iterate. If shorter, pad.
            target_len = 15600
            
            if len(audio) < target_len:
                # Pad with zeros
                pad_width = target_len - len(audio)
                audio = np.pad(audio, (0, pad_width), mode='constant')
            elif len(audio) > target_len:
                # Take the first chunk for now (or loop for better accuracy)
                audio = audio[:target_len]

            # Run inference
            # Input name for YAMNet ONNX is usually 'waveform'
            inputs = {self.session.get_inputs()[0].name: audio}
            
            # Outputs: [prediction, embedding, log_mel_spectrogram]
            outputs = self.session.run(None, inputs)
            
            # Prediction scores (logits or probabilities? usually probabilities)
            scores = outputs[0][0] # First batch
            
            # Get top K
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for i in top_indices:
                results.append((self.class_names[i], float(scores[i])))
                
            return results

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return []
