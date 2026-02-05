import time
from typing import Dict, Any
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

class CheckBabyCrying(Tool):
    """Check if a baby cry was recently detected."""

    name = "check_baby_crying"
    description = "Check if the system has recently heard a baby crying. Use this when the user asks 'Is the baby crying?' or 'Do you hear anything?'."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies) -> Dict[str, Any]:
        """Check status of audio classifier."""
        if not deps.audio_classifier_status:
            return {"status": "error", "message": "Audio classifier status not available."}
        
        last_event = deps.audio_classifier_status.get("latest_event")
        last_time = deps.audio_classifier_status.get("timestamp", 0)
        
        now = time.time()
        time_diff = now - last_time
        
        if last_event and time_diff < 30: # Within last 30 seconds
            return {
                "status": "crying_detected", 
                "message": f"Yes, I detected {last_event} {int(time_diff)} seconds ago.",
                "event": last_event,
                "seconds_ago": int(time_diff)
            }
        else:
            return {
                "status": "quiet", 
                "message": "I haven't heard any crying recently."
            }
