"""Tool to send a photo via Signal."""

import logging
import tempfile
import os
from typing import Any, Dict

import cv2

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.send_signal import get_signal_interface
from reachy_mini_conversation_app.config import config

logger = logging.getLogger(__name__)


class SendSignalPhotoTool(Tool):
    """Take a photo and send it via Signal."""

    name = "send_signal_photo"
    description = "Take a photo and send it to the user's phone via Signal."
    parameters_schema = {
        "type": "object",
        "properties": {
            "caption": {
                "type": "string",
                "description": "Optional caption for the photo.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a photo and send it via Signal."""
        message = kwargs.get("caption") or "Photo from Reachy Mini"
        phone_number = config.SIGNAL_USER_PHONE

        if not phone_number:
            return {"success": False, "error": "No phone number provided and no default configured"}

        # Get frame from camera
        if deps.camera_worker is None:
            return {"success": False, "error": "Camera not available"}

        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            return {"success": False, "error": "No frame available from camera"}

        signal = get_signal_interface()
        if not signal.available:
            return {"success": False, "error": "Signal CLI not available"}

        # Save frame to temp file
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                temp_file = f.name
                success, buffer = cv2.imencode('.jpg', frame)
                if not success:
                    return {"success": False, "error": "Failed to encode image"}
                f.write(buffer.tobytes())

            # Send via Signal with attachment
            success = await signal.send_message(message, phone_number, attachment=temp_file)

            if success:
                logger.info(f"Signal photo sent to {phone_number}")
                return {"success": True, "sent_to": phone_number}
            else:
                return {"success": False, "error": "Failed to send photo"}

        except Exception as e:
            logger.error(f"Signal photo error: {e}")
            return {"success": False, "error": str(e)}

        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
