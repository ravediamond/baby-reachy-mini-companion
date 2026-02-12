"""Camera worker thread with frame buffering.

Provides:
- 30Hz+ camera polling with thread-safe frame buffering
- Latest frame always available for tools
"""

import time
import logging
import threading

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini


logger = logging.getLogger(__name__)


class CameraWorker:
    """Thread-safe camera worker with frame buffering."""

    def __init__(self, reachy_mini: ReachyMini) -> None:
        """Initialize."""
        self.reachy_mini = reachy_mini

        # Thread-safe frame storage
        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            # Return a copy in original BGR format (OpenCV native)
            return self.latest_frame.copy()

    def start(self) -> None:
        """Start the camera worker loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        logger.debug("Camera worker stopped")

    def working_loop(self) -> None:
        """Poll frames from the camera and store the latest one."""
        logger.debug("Starting camera working loop")

        while not self._stop_event.is_set():
            try:
                # Get frame from robot
                frame = self.reachy_mini.media.get_frame()

                if frame is not None:
                    # Thread-safe frame storage
                    with self.frame_lock:
                        self.latest_frame = frame

                # Small sleep to prevent excessive CPU usage
                time.sleep(0.04)

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)  # Longer sleep on error

        logger.debug("Camera worker thread exited")
