import logging
import asyncio
import numpy as np
from smolagents import Tool
from typing import Optional

logger = logging.getLogger(__name__)

class SpeakTool(Tool):
    name = "speak_to_room"
    description = "Speaks text out loud to the physical room using the robot's speakers. Use this when the user is near the robot or for general announcements."
    inputs = {
        "text": {
            "type": "string",
            "description": "The text to speak out loud."
        },
        "emotion": {
            "type": "string",
            "description": "Optional emotion tag (e.g., 'happy', 'sad') to influence voice tone.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, tts, output_queue, head_wobbler=None):
        super().__init__()
        self.tts = tts
        self.output_queue = output_queue
        self.head_wobbler = head_wobbler

    def forward(self, text: str, emotion: Optional[str] = None) -> str:
        # Since smolagents runs tools in a thread, we need to schedule the async task
        # back onto the main event loop if possible, or run a new loop.
        # But here, we just need to push to a queue or run synthesis.
        
        # We can use asyncio.run_coroutine_threadsafe if we have reference to the main loop
        # For now, let's assume we can block briefly or use a fire-and-forget strategy via run_coroutine_threadsafe
        
        # IMPORTANT: This tool is running inside the Agent's execution thread, 
        # but the TTS/Audio system is likely on the main asyncio loop.
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # We are likely in a separate thread.
            # We need to find the main loop. 
            # HACK: We assume the loop was attached to the tool or we can't do async properly.
            # Let's try to just use run_coroutine_threadsafe against the loop stored in the handler.
            # But we only passed tts/queue.
            pass

        # We need the loop. Let's assume it's attached to the TTS object or we pass it.
        # Let's attach it in __init__
        pass 
        return "Spoke: " + text

    # Re-implementing correctly with loop handling
    def set_loop(self, loop):
        self.loop = loop

    def forward(self, text: str, emotion: Optional[str] = None) -> str:
        if not hasattr(self, 'loop'):
            return "Error: Async loop not connected."
        
        # Fire and forget synthesis task
        asyncio.run_coroutine_threadsafe(self._speak(text, emotion), self.loop)
        return f"Speaking to room: '{text}'"

    async def _speak(self, text: str, emotion: str = None):
        if not self.tts:
            return
            
        speed = 1.0
        voice = "af_sarah"
        
        if emotion == "happy":
            speed = 1.1
        
        try:
            # Synthesize
            sr, audio = await self.tts.synthesize(text, voice=voice, speed=speed)
            
            # Convert
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Output
            await self.output_queue.put((24000, audio_int16.reshape(1, -1)))
            
            # Wobbler
            if self.head_wobbler:
                 import base64
                 b64_data = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
                 self.head_wobbler.feed(b64_data)
                 
        except Exception as e:
            logger.error(f"SpeakTool error: {e}")


class SignalTool(Tool):
    name = "send_signal_text"
    description = "Sends a text message to the user's phone via Signal. Use this for notifications, lists, or when the user is away."
    inputs = {
        "text": {
            "type": "string",
            "description": "The message content to send."
        },
        "recipient": {
            "type": "string",
            "description": "The phone number to send to. If unknown, leave blank to use default.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, signal_interface, default_recipient=None):
        super().__init__()
        self.signal = signal_interface
        self.default_recipient = default_recipient

    def set_loop(self, loop):
        self.loop = loop

    def forward(self, text: str, recipient: Optional[str] = None) -> str:
        if not hasattr(self, 'loop'):
            return "Error: Async loop not connected."
            
        target = recipient or self.default_recipient
        if not target:
            return "Error: No recipient specified."

        asyncio.run_coroutine_threadsafe(self.signal.send_message(text, target), self.loop)
        return f"Sent Signal message to {target}"
