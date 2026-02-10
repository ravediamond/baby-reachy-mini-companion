"""YOLO-based dangerous object detector for baby safety.

Uses a general-purpose YOLO model (COCO classes) to detect objects
that could be hazardous near a baby — scissors, knives, forks, etc.
When a dangerous object is detected the handler triggers a VLM
analysis and alerts the parent via Signal.
"""

from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


try:
    from ultralytics import YOLO  # type: ignore
except ImportError as e:
    raise ImportError(
        "To use the danger detector, please install the extra dependencies: pip install '.[yolo_vision]'",
    ) from e


logger = logging.getLogger(__name__)


# COCO class names considered dangerous for a baby
DANGEROUS_OBJECTS: set[str] = {
    "scissors",
    "knife",
    "fork",
}


class DangerDetector:
    """Detect dangerous objects in camera frames using YOLO."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Initialize the danger detector with a YOLO model."""
        self.confidence_threshold = confidence_threshold
        try:
            self.model = YOLO(model_name).to(device)
            logger.info(f"Danger detector loaded ({model_name} on {device})")
        except Exception as e:
            logger.error(f"Failed to load danger detection model: {e}")
            raise

    def detect(self, frame: NDArray[np.uint8]) -> List[Dict[str, object]]:
        """Run YOLO inference and return any dangerous objects found.

        Returns:
            List of dicts with keys: label, confidence, bbox.
            Empty list when nothing dangerous is detected.

        """
        try:
            results = self.model(frame, verbose=False)
        except Exception as e:
            logger.error(f"Danger detection inference error: {e}")
            return []

        dangerous: List[Dict[str, object]] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                conf = float(box.conf[0])

                if label in DANGEROUS_OBJECTS and conf >= self.confidence_threshold:
                    dangerous.append(
                        {
                            "label": label,
                            "confidence": round(conf, 2),
                            "bbox": box.xyxy[0].tolist(),
                        }
                    )

        return dangerous
