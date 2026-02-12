"""YOLO-based dangerous object detector for baby safety.

Uses a general-purpose YOLO model (COCO classes) to detect objects
that could be hazardous near a baby — scissors, knives, forks, etc.
When a dangerous object is detected the handler triggers a VLM
analysis and alerts the parent via Signal.

Detection uses multi-frame confirmation: a dangerous object must appear
in at least ``confirm_frames`` of the last ``window_size`` frames before
an alert is raised.  This filters out single-frame hallucinations that
small YOLO models produce in dim lighting (e.g. "dog", "umbrella").

Default model is ``yolo26n`` (YOLO v26 nano) which offers better
small-object accuracy and 43% faster CPU inference vs yolo11s —
important for Jetson deployment.
"""

from __future__ import annotations
import logging
from collections import deque
from typing import Dict, List, Set

import numpy as np
from numpy.typing import NDArray


try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "Danger detector requires 'ultralytics'. Re-run: uv sync",
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
        model_name: str = "yolo26n.pt",
        confidence_threshold: float = 0.2,
        confirm_frames: int = 3,
        window_size: int = 5,
        device: str = "cpu",
    ) -> None:
        """Initialize the danger detector with a YOLO model.

        Args:
            model_name: YOLO model to use (yolo26n recommended).
            confidence_threshold: Minimum confidence to consider a detection.
            confirm_frames: Number of frames an object must appear in to
                be confirmed (within the sliding window).
            window_size: Size of the sliding window (number of recent frames).
            device: Inference device ('cpu' or 'cuda').

        """
        self.confidence_threshold = confidence_threshold
        self.confirm_frames = confirm_frames
        self._detection_history: deque[Set[str]] = deque(maxlen=window_size)
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
        all_labels: List[str] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                conf = float(box.conf[0])
                all_labels.append(f"{label}({conf:.2f})")

                if label in DANGEROUS_OBJECTS and conf >= self.confidence_threshold:
                    dangerous.append(
                        {
                            "label": label,
                            "confidence": round(conf, 2),
                            "bbox": box.xyxy[0].tolist(),
                        }
                    )

        if all_labels:
            logger.info(f"YOLO detections: {', '.join(all_labels)}")

        if dangerous:
            labels = ", ".join(f"{d['label']}({d['confidence']})" for d in dangerous)
            logger.info(f"Dangerous objects spotted: {labels}")

        return dangerous

    def detect_confirmed(self, frame: NDArray[np.uint8]) -> List[Dict[str, object]]:
        """Detect dangerous objects with multi-frame confirmation.

        Runs single-frame detection, records which dangerous labels were
        found, and returns only those that have appeared in at least
        ``confirm_frames`` of the last ``window_size`` frames.

        Returns:
            List of confirmed dangerous detections (same format as detect()).

        """
        detections = self.detect(frame)

        # Record which dangerous labels appeared in this frame
        frame_dangers: Set[str] = {str(d["label"]) for d in detections}
        self._detection_history.append(frame_dangers)

        # Count appearances in the sliding window
        confirmed: List[Dict[str, object]] = []
        seen_labels: Set[str] = set()
        for det in detections:
            label = str(det["label"])
            if label in seen_labels:
                continue
            count = sum(1 for past in self._detection_history if label in past)
            if count >= self.confirm_frames:
                confirmed.append(det)
                seen_labels.add(label)
                logger.info(
                    f"Danger CONFIRMED: {label} seen in {count}/{len(self._detection_history)} recent frames"
                )

        return confirmed
