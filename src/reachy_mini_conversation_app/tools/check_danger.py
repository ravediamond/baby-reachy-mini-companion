"""Tool to check if dangerous objects were recently detected near the baby."""

import time
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class CheckDanger(Tool):
    """Check if the vision safety monitor has detected any dangerous objects."""

    name = "check_danger"
    description = (
        "Check if any dangerous objects (scissors, knives, forks, etc.) "
        "were recently detected near the baby by the vision safety monitor."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return the latest danger detection status."""
        if deps.vision_threat_status is None:
            return {"status": "unavailable", "message": "Vision safety monitor not active"}

        latest = deps.vision_threat_status.get("latest_threat")
        ts = deps.vision_threat_status.get("timestamp", 0.0)
        now = time.time()

        if latest and (now - ts) < 60:
            return {
                "status": "danger_detected",
                "threat": latest,
                "objects": deps.vision_threat_status.get("objects", []),
                "seconds_ago": int(now - ts),
            }

        return {"status": "safe", "message": "No dangerous objects detected recently"}
