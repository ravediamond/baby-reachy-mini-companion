import asyncio
import logging
import numpy as np
from typing import Tuple, Optional, Literal
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, dispatch_tool_call
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.local.vad import SileroVAD
from reachy_mini_conversation_app.local.stt import LocalSTT
from reachy_mini_conversation_app.local.llm import LocalLLM
from reachy_mini_conversation_app.local.tts import LocalTTS

logger = logging.getLogger(__name__)

class LocalSessionHandler(AsyncStreamHandler):
    """Local processing handler for Reachy Mini (VAD -> STT -> LLM -> TTS)."""

    def __init__(self, deps: ToolDependencies, llm_url: str = "http://localhost:11434/v1", llm_model: str = "ministral-3b"):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24000,
            input_sample_rate=16000, 
        )
        self.deps = deps
        self.llm_url = llm_url
        self.llm_model = llm_model
        
        self.output_queue = asyncio.Queue()
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        
        # Initialize models lazily or in start_up
        self.vad = None
        self.stt = None
        self.llm = None
        self.tts = None
        self.pipeline_task = None

    async def start_up(self):
        """Initialize local models."""
        logger.info("Initializing Local AI Pipeline...")
        
        self.vad = SileroVAD()
        self.stt = LocalSTT(model_size="base.en") # Adjustable
        self.llm = LocalLLM(base_url=self.llm_url, model=self.llm_model)
        self.tts = LocalTTS()
        
        # Load system prompt
        self.llm.set_system_prompt(get_session_instructions())
        
        logger.info("Local Pipeline Ready.")

    async def receive(self, frame: Tuple[int, np.ndarray]) -> None:
        """Process incoming audio frame."""
        _, audio = frame
        
        # Accumulate buffer
        # fastrtc gives int16, we need float32 for VAD/STT often
        # But Silero VAD expects float32.
        
        # Convert to float32 normalized
        audio_float = audio.astype(np.float32) / 32768.0
        
        if audio_float.ndim > 1:
            audio_float = audio_float.flatten()
            
        self.audio_buffer.append(audio_float)
        
        # VAD Logic (Simplified)
        # Check every N frames
        if len(self.audio_buffer) % 5 == 0:
            # Check last chunk
            is_speech = self.vad.is_speech(audio_float)
            
            if is_speech:
                self.is_speaking = True
                self.silence_frames = 0
                if self.deps.head_wobbler:
                    self.deps.movement_manager.set_listening(True)
            else:
                if self.is_speaking:
                    self.silence_frames += 1
                    
                    # Silence threshold (e.g. 0.5s ~ 25 frames at 20ms chunks)
                    # This logic depends on chunk size coming from fastrtc
                    if self.silence_frames > 20: 
                        # End of speech detected
                        self.is_speaking = False
                        self.silence_frames = 0
                        self.deps.movement_manager.set_listening(False)
                        
                        # Trigger pipeline
                        full_audio = np.concatenate(self.audio_buffer)
                        self.audio_buffer = [] # Clear buffer
                        
                        asyncio.create_task(self._run_pipeline(full_audio))

    async def _run_pipeline(self, audio: np.ndarray):
        """Run STT -> LLM -> TTS pipeline."""
        
        # 1. STT
        transcript = await asyncio.to_thread(self.stt.transcribe, audio)
        if not transcript.strip():
            return
            
        logger.info(f"User: {transcript}")
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
        
        # 2. LLM
        full_response_text = ""
        current_sentence = ""
        
        async for token in self.llm.chat_stream(transcript):
            full_response_text += token
            current_sentence += token
            
            # Simple sentence splitting for TTS streaming
            if token in [".", "!", "?", "\n"]:
                # 3. TTS (Streaming)
                await self._process_sentence(current_sentence)
                current_sentence = ""
        
        # Process remaining text
        if current_sentence.strip():
            await self._process_sentence(current_sentence)
            
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": full_response_text}))

    async def _process_sentence(self, text: str):
        """Synthesize and queue audio for a sentence, handling tone."""
        text = text.strip()
        if not text:
            return

        # Simple Tone Analysis
        speed = 1.0
        voice = "af_sarah" # Default cheerful
        
        if "[HAPPY]" in text:
            text = text.replace("[HAPPY]", "")
            speed = 1.1
            # self.deps.movement_manager... trigger emotion?
            
        if "[STORY]" in text:
            text = text.replace("[STORY]", "")
            speed = 0.9
            
        # Synthesize
        sr, audio = await self.tts.synthesize(text, voice=voice, speed=speed)
        
        # Convert back to int16 for output
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Resample if needed (Kokoro is usually 24k, we output 24k)
        if sr != 24000:
            # Resample logic here if needed
            pass
            
        # Send to speaker
        await self.output_queue.put((24000, audio_int16.reshape(1, -1)))
        
        # Feed head wobbler
        # Need base64 delta? HeadWobbler expects bytes or something?
        # Openai handler sent bytes.
        # We can adapt HeadWobbler or just feed it directly if it accepts numpy?
        # OpenaiRealtimeHandler: self.deps.head_wobbler.feed(event.delta) -> base64
        # We might need to mock that or update HeadWobbler.
        
        if self.deps.head_wobbler:
             # Convert to base64 to match existing HeadWobbler interface
             import base64
             b64_data = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
             self.deps.head_wobbler.feed(b64_data)

    async def emit(self):
        return await wait_for_item(self.output_queue)

    async def shutdown(self):
        pass
