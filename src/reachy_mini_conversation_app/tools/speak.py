from typing import Dict, Any
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

class Speak(Tool):
    """Tool for the robot to speak a given text."""

    name = "speak"
    description = "Make the robot speak the provided text out loud. Use this when you want to initiate a verbal statement or respond while performing other actions."
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to speak."
            },
            "emotion": {
                "type": "string",
                "description": "Optional emotion to express while speaking (e.g., 'happy', 'sad')."
            }
        },
        "required": ["text"]
    }

    async def __call__(self, deps: ToolDependencies, text: str, emotion: str = None) -> Dict[str, Any]:
        """Execute the speak tool."""
        
        # If emotion is provided, we could optionally trigger play_emotion here or append it to text
        full_text = text
        if emotion:
            full_text = f"[{emotion.upper()}] {text}"

        if deps.speak_func:
            # speak_func is expected to be an async function
            await deps.speak_func(full_text)
            return {"status": "success", "message": f"Spoke: {text}"}
        else:
            return {"status": "error", "message": "Speech function not available."}
