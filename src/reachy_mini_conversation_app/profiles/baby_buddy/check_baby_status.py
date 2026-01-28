import logging
import base64
import cv2
import asyncio
from typing import Any, Dict
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

class CheckBabyStatus(Tool):
    """Checks the baby's status using the camera (Sentinel Mode)."""

    name = "check_baby_status"
    description = "Captures an image to check if the baby is safe, crying, or in danger."
    parameters_schema = {
        "type": "object",
        "properties": {},  # No arguments needed, the intent is hardcoded
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture and specifically analyze it for safety."""
        logger.info("Sentinel Check: Checking baby status...")

        # 1. Capture Frame
        if deps.camera_worker is None:
            return {"error": "Camera not available"}
        
        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            return {"error": "No frame captured"}

        # 2. Define the Safety Question
        safety_query = (
            "Analyze this image specifically for a baby's safety. "
            "Is the baby crying, in an unsafe position, or is the face not visible? "
            "Reply strictly with 'SAFE' if everything is okay, or a short description of the problem if not."
        )

        # 3. Analyze (Local Vision or Cloud Vision)
        if deps.vision_manager is not None:
            # Local Vision (SmolVLM2)
            result = await asyncio.to_thread(
                deps.vision_manager.processor.process_image, frame, safety_query
            )
            return {"status": result}
        else:
            # GPT-Realtime Vision (Cloud)
            # We return the image to the LLM with the specific prompt context
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                return {"error": "Failed to encode frame"}
            
            b64_im = base64.b64encode(buffer.tobytes()).decode("utf-8")
            
            # The LLM will receive this image and answer the prompt defined in the tool description/context
            return {
                "b64_im": b64_im, 
                "system_note": "Image captured. Analyze it: " + safety_query
            }