"""Tool to send Signal messages."""

import logging
import asyncio
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.input.signal_interface import SignalInterface
from reachy_mini_conversation_app.config import config

logger = logging.getLogger(__name__)

# Shared Signal interface instance
_signal_interface = None


def get_signal_interface() -> SignalInterface:
    """Get or create the shared Signal interface."""
    global _signal_interface
    if _signal_interface is None:
        _signal_interface = SignalInterface()
    return _signal_interface


class SendSignalTool(Tool):
    """Send a message via Signal."""

    name = "send_signal"
    description = "Send a text message to the user's phone via Signal. Default recipient is already configured - just provide the message."
    parameters_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to send.",
            },
        },
        "required": ["message"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Send a Signal message."""
        message = kwargs.get("message", "")
        phone_number = kwargs.get("phone_number") or config.SIGNAL_USER_PHONE

        if not message:
            return {"success": False, "error": "No message provided"}

        if not phone_number:
            return {"success": False, "error": "No phone number provided and no default configured"}

        signal = get_signal_interface()

        if not signal.available:
            return {"success": False, "error": "Signal CLI not available"}

        try:
            success = await signal.send_message(message, phone_number)
            if success:
                logger.info(f"Signal message sent to {phone_number}: {message[:50]}...")
                return {"success": True, "sent_to": phone_number}
            else:
                return {"success": False, "error": "Failed to send message"}
        except Exception as e:
            logger.error(f"Signal send error: {e}")
            return {"success": False, "error": str(e)}
