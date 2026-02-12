from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove


class SootheBaby(Tool):
    """Soothe a baby with gentle words and rocking motions."""

    name = "soothe_baby"
    description = "Calm and soothe a baby. Plays a gentle lullaby message and performs slow, rocking movements."
    parameters_schema = {
        "type": "object",
        "properties": {
            "duration_seconds": {
                "type": "integer",
                "description": "Approximate duration of the soothing session (default 10).",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute the soothing routine."""
        duration_seconds: int = int(kwargs.get("duration_seconds", 10))
        # 1. gentle rocking motions
        moves = ["side_to_side_sway", "pendulum_swing"]

        # Queue moves
        if deps.movement_manager:
            # Queue a few gentle moves to last roughly the duration
            # Each move is typically 2-4 seconds.
            count = max(1, duration_seconds // 3)
            for i in range(count):
                move_name = moves[i % len(moves)]
                dance_move = DanceQueueMove(move_name)
                deps.movement_manager.queue_move(dance_move)

        # 2. Soothing speech (using [STORY] for slower pace)
        soothing_text = (
            "[STORY] Hush now... it's okay... [sad] sleep tight little one... "
            "[STORY] everything is calm... safe and sound... hush little baby..."
        )

        if deps.speak_func:
            await deps.speak_func(soothing_text)
            return {"status": "success", "message": "Soothing baby with gentle rocking and lullaby."}
        else:
            return {"status": "partial_success", "message": "Rocking, but cannot speak (no speak_func)."}
