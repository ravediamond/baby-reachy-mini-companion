import asyncio
import logging
import re
import json
import numpy as np
from typing import Tuple, Optional, Literal, List, Dict, Any
from scipy.signal import resample
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, dispatch_tool_call, get_tool_specs
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.local.vad import SileroVAD
from reachy_mini_conversation_app.local.stt import LocalSTT
from reachy_mini_conversation_app.local.llm import LocalLLM
from reachy_mini_conversation_app.local.tts import LocalTTS

logger = logging.getLogger(__name__)

class LocalSessionHandler(AsyncStreamHandler):
    """Local processing handler for Reachy Mini (VAD -> STT -> LLM -> TTS)."""

    def __init__(self, deps: ToolDependencies, llm_url: str = "http://localhost:11434/v1", llm_model: str = None):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24000,
            input_sample_rate=16000, 
        )
        self.deps = deps
        self.llm_url = llm_url
        self.llm_model = llm_model or config.LOCAL_LLM_MODEL or "qwen2.5:3b" # Fallback if env is missing
        
        self.output_queue = asyncio.Queue()
        self.audio_buffer = [] # Accumulates 16k chunks for STT
        self.vad_buffer = np.array([], dtype=np.float32) # Accumulates for VAD windowing
        
        self.is_speaking = False
        self.silence_chunks = 0
        self.speech_chunks = 0
        
        # Initialize models lazily or in start_up
        self.vad = None
        self.stt = None
        self.llm = None
        self.tts = None
        self.pipeline_task = None
        
        # Tool specs cache
        self.tool_specs = get_tool_specs()

    def copy(self) -> "LocalSessionHandler":
        """Create a copy of the handler."""
        return LocalSessionHandler(self.deps, self.llm_url, self.llm_model)

    async def start_up(self):
        """Initialize local models."""
        logger.info("Initializing Local AI Pipeline...")
        
        try:
            logger.info("Loading VAD (Silero)...")
            self.vad = await asyncio.to_thread(SileroVAD)
            
            logger.info(f"Loading STT (Faster-Whisper {config.LOCAL_STT_MODEL})...")
            self.stt = await asyncio.to_thread(LocalSTT, model_size=config.LOCAL_STT_MODEL)
            
            logger.info("Loading LLM Client...")
            self.llm = LocalLLM(base_url=self.llm_url, model=self.llm_model)
            
            logger.info("Loading TTS (Kokoro)...")
            self.tts = await asyncio.to_thread(LocalTTS)
            
            # Load system prompt
            logger.info("Setting system prompt...")
            self.llm.set_system_prompt(get_session_instructions())
            
            logger.info("Local Pipeline Ready.")
        except Exception as e:
            logger.error(f"Failed to initialize local pipeline: {e}")
            raise

    async def receive(self, frame: Tuple[int, np.ndarray]) -> None:
        """Process incoming audio frame."""
        sr, audio = frame
        
        # 1. Convert to float32 safely
        if audio.dtype == np.float32:
            audio_float = audio.copy()
        else:
            audio_float = audio.astype(np.float32) / 32768.0
        
        # 2. Downmix to mono if needed
        if audio_float.ndim > 1:
            audio_float = np.mean(audio_float, axis=1)

        # 3. Resample to 16000 Hz if needed (Silero requires 16k)
        target_sr = 16000
        if sr != target_sr:
            num_samples = int(len(audio_float) * target_sr / sr)
            audio_float = resample(audio_float, num_samples)
            
        # 4. Add to VAD buffer
        self.vad_buffer = np.concatenate((self.vad_buffer, audio_float))
        
        # 5. Process VAD in 512-sample chunks (32ms at 16k)
        chunk_size = 512
        
        while len(self.vad_buffer) >= chunk_size:
            chunk = self.vad_buffer[:chunk_size]
            self.vad_buffer = self.vad_buffer[chunk_size:]
            
            # VAD Check
            if self.vad:
                is_speech = self.vad.is_speech(chunk)
            else:
                is_speech = False
            
            if is_speech:
                if not self.is_speaking:
                    # Speech started
                    self.is_speaking = True
                    self.speech_chunks = 0
                    logger.info("Speech detected!")
                    if self.deps.head_wobbler:
                        self.deps.movement_manager.set_listening(True)
                
                self.silence_chunks = 0
                self.speech_chunks += 1
                self.audio_buffer.append(chunk)
                
            else:
                if self.is_speaking:
                    # Possibly speech ended, but wait for silence threshold
                    self.silence_chunks += 1
                    self.audio_buffer.append(chunk)
                    
                    # Silence threshold: 0.8s ~ 25 chunks (25 * 32ms = 800ms)
                    if self.silence_chunks > 25:
                        self.is_speaking = False
                        logger.info(f"Speech finished ({self.speech_chunks} chunks)")
                        self.deps.movement_manager.set_listening(False)
                        
                        # Trigger pipeline if we had enough speech
                        if self.speech_chunks > 10: # Avoid noise blips
                            full_audio = np.concatenate(self.audio_buffer)
                            logger.info("Triggering AI pipeline...")
                            asyncio.create_task(self._run_pipeline(full_audio))
                        else:
                            logger.debug("Speech too short, ignoring.")
                        
                        self.audio_buffer = []
                        self.speech_chunks = 0
                else:
                    # Not speaking, just silence. 
                    pass

    async def _run_pipeline(self, audio: np.ndarray):
        """Run STT -> LLM -> TTS pipeline."""
        
        # 1. STT
        transcript = await asyncio.to_thread(self.stt.transcribe, audio)
        if not transcript.strip():
            return

        # 1.5 Inject Vision Context
        vision_context = ""
        if self.deps.vision_manager and self.deps.vision_manager.latest_description:
            vision_context = f" [Visual Context: {self.deps.vision_manager.latest_description}]"
            
        logger.info(f"User: {transcript}{vision_context}")
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
        
        # 2. LLM Loop (Handling potential tool calls)
        current_input = transcript + vision_context
        max_turns = 3  # Prevent infinite loops
        turn = 0
        
        tool_outputs = None # For subsequent turns

        while turn < max_turns:
            turn += 1
            full_response_text = ""
            current_sentence = ""
            
            # If this is the first turn, we pass user input. Subsequent turns are tool outputs.
            llm_user_input = current_input if turn == 1 else None
            
            logger.info(f"LLM Turn {turn} (Input: {llm_user_input or 'Tool Outputs'})")
            
            tool_calls = []

            async for event in self.llm.chat_stream(user_text=llm_user_input, tools=self.tool_specs, tool_outputs=tool_outputs):
                
                if event["type"] == "text":
                    token = event["content"]
                    full_response_text += token
                    current_sentence += token
                    
                    # Simple sentence splitting for TTS streaming
                    if token in [".", "!", "?", "\n"]:
                        await self._process_sentence(current_sentence)
                        current_sentence = ""
                
                elif event["type"] == "tool_call":
                    tool_calls.append(event["tool_call"])
                
                elif event["type"] == "error":
                    logger.error(f"LLM Error: {event['content']}")

            # Process remaining text
            if current_sentence.strip():
                await self._process_sentence(current_sentence)
            
            if full_response_text:
                logger.info(f"Assistant: {full_response_text}")
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": full_response_text}))

            # If no tool calls, we are done
            if not tool_calls:
                break
            
            # Execute Tools
            tool_outputs = []
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                args_str = tc["function"]["arguments"]
                call_id = tc["id"]
                
                logger.info(f"🛠️ Tool Call: {func_name}({args_str})")
                
                # Execute
                result = await dispatch_tool_call(func_name, args_str, self.deps)
                result_str = json.dumps(result)
                
                tool_outputs.append({
                    "role": "tool", 
                    "content": result_str,
                    "tool_call_id": call_id
                })
                
                logger.info(f"   -> Result: {result_str}")
            
            # Prepare for next turn
            # We Loop back with tool_outputs, llm_user_input will be None
        
    async def _process_sentence(self, text: str):
        """Synthesize and queue audio for a sentence, handling tone and embedded commands."""
        original_text = text
        
        # --- 1. Extract and Execute Commands (Fallback / Legacy) ---
        # Look for PLAY_EMOTION("emotion_name") - Kept for backward compatibility or if LLM hallucinates text
        emotion_matches = re.findall(r'play_emotion\s*\(\s*["\']([^"\']+)["\']\s*\)', text, re.IGNORECASE)
        for emotion in emotion_matches:
            logger.info(f"⚡️ Executing implied command (Legacy): play_emotion('{emotion}')")
            asyncio.create_task(dispatch_tool_call("play_emotion", f'{{"emotion_name": "{emotion}"}}', self.deps))

        # --- 2. Clean Text for TTS ---
        # Remove the command strings we just found (more robust regex)
        text = re.sub(r'\(?play_emotion\s*\([^)]+\)\)?', '', text, flags=re.IGNORECASE)
        
        # Remove bold/italic markdown (*word*, **word**)
        text = text.replace("**", "").replace("*", "")

        # Remove action descriptions in parentheses or asterisks if they remain
        # e.g. (waves hands) or *laughs*
        text = re.sub(r'\s*\([^)]*\)', '', text)
        
        # Remove other common artifacts if any
        text = text.strip()
        
        if not text:
            return

        # Simple Tone Analysis
        speed = 1.0
        voice = "af_sarah" # Default cheerful
        
        if "[HAPPY]" in text:
            text = text.replace("[HAPPY]", "")
            speed = 1.1
            
        if "[STORY]" in text:
            text = text.replace("[STORY]", "")
            speed = 0.9
            
        # Synthesize
        sr, audio = await self.tts.synthesize(text, voice=voice, speed=speed)
        
        # Convert back to int16 for output
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Send to speaker
        await self.output_queue.put((24000, audio_int16.reshape(1, -1)))
        
        # Feed head wobbler
        if self.deps.head_wobbler:
             import base64
             b64_data = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
             self.deps.head_wobbler.feed(b64_data)

    async def emit(self):
        return await wait_for_item(self.output_queue)

    async def shutdown(self):
        pass
    async def emit(self):
        return await wait_for_item(self.output_queue)

    async def shutdown(self):
        pass
