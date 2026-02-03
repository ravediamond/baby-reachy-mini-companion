import asyncio
import logging
import numpy as np
from typing import Tuple, Optional, Any
from scipy.signal import resample
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item

from smolagents import CodeAgent, LiteLLMModel

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.local.vad import SileroVAD
from reachy_mini_conversation_app.local.stt import LocalSTT
from reachy_mini_conversation_app.local.tts import LocalTTS

from reachy_mini_conversation_app.input.signal_interface import SignalInterface
from reachy_mini_conversation_app.tools.omni_tools import SpeakTool, SignalTool

logger = logging.getLogger(__name__)

class OmniSessionHandler(AsyncStreamHandler):
    """Omni-Channel Agent Handler (Audio + Signal)."""

    def __init__(self, deps: ToolDependencies, llm_model: str = "ollama_chat/llama3.2"):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24000,
            input_sample_rate=16000, 
        )
        self.deps = deps
        self.llm_model_id = llm_model 
        
        self.output_queue = asyncio.Queue()
        self.audio_buffer = [] 
        self.vad_buffer = np.array([], dtype=np.float32)
        
        self.is_speaking = False
        self.silence_chunks = 0
        self.speech_chunks = 0
        
        # Components
        self.vad = None
        self.stt = None
        self.tts = None
        self.signal = None
        self.agent = None
        
        self.loop = None
        self.polling_task = None
        
        # User Config (Hardcoded for prototype, move to env/config later)
        self.user_phone = "+33658273673" # Replace with actual number or env var

    async def start_up(self):
        """Initialize models and agent."""
        self.loop = asyncio.get_running_loop()
        logger.info("Initializing Omni-Channel Agent...")
        
        try:
            # 1. Audio Models
            self.vad = await asyncio.to_thread(SileroVAD)
            self.stt = await asyncio.to_thread(LocalSTT, model_size=config.LOCAL_STT_MODEL)
            self.tts = await asyncio.to_thread(LocalTTS)
            
            # 2. Signal Interface
            self.signal = SignalInterface()
            
            # 3. Agent Tools
            self.speak_tool = SpeakTool(self.tts, self.output_queue, self.deps.head_wobbler)
            self.speak_tool.set_loop(self.loop)
            
            self.signal_tool = SignalTool(self.signal, default_recipient=self.user_phone)
            self.signal_tool.set_loop(self.loop)
            
            # 4. The Brain (Smolagents)
            # Using LiteLLMModel to connect to Ollama
            model = LiteLLMModel(
                model_id=self.llm_model_id,
                api_base="http://localhost:11434", 
                api_key="ollama" 
            )
            
            self.agent = CodeAgent(
                tools=[self.speak_tool, self.signal_tool], 
                model=model,
                add_base_tools=True # Adds python interpreter, DuckDuckGo etc if available
            )
            
            # 5. Start Signal Poller
            self.polling_task = asyncio.create_task(self._poll_signal())
            
            logger.info("Omni-Channel Agent Ready.")
            
        except Exception as e:
            logger.error(f"Failed to initialize omni agent: {e}")
            raise

    async def _poll_signal(self):
        """Background task to poll Signal messages."""
        logger.info("Starting Signal Poller...")
        while True:
            try:
                messages = await self.signal.poll_messages()
                for msg in messages:
                    sender = msg["sender"]
                    text = msg["content"]
                    logger.info(f"Signal received from {sender}: {text}")
                    
                    # Feed to Agent
                    prompt = f"[SOURCE: SIGNAL] User {sender} wrote: {text}"
                    await self._run_agent(prompt)
                    
                await asyncio.sleep(2.0) # Poll interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(5.0)

    async def receive(self, frame: Tuple[int, np.ndarray]) -> None:
        """Process incoming audio frame (VAD logic)."""
        sr, audio = frame
        
        # Basic conversion (same as LocalSessionHandler)
        if audio.dtype == np.float32:
            audio_float = audio.copy()
        else:
            audio_float = audio.astype(np.float32) / 32768.0
        
        if audio_float.ndim > 1:
            audio_float = np.mean(audio_float, axis=1)

        target_sr = 16000
        if sr != target_sr:
            num_samples = int(len(audio_float) * target_sr / sr)
            audio_float = resample(audio_float, num_samples)
            
        self.vad_buffer = np.concatenate((self.vad_buffer, audio_float))
        chunk_size = 512
        
        while len(self.vad_buffer) >= chunk_size:
            chunk = self.vad_buffer[:chunk_size]
            self.vad_buffer = self.vad_buffer[chunk_size:]
            
            is_speech = self.vad.is_speech(chunk) if self.vad else False
            
            if is_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_chunks = 0
                    if self.deps.head_wobbler:
                        self.deps.movement_manager.set_listening(True)
                
                self.silence_chunks = 0
                self.speech_chunks += 1
                self.audio_buffer.append(chunk)
                
            else:
                if self.is_speaking:
                    self.silence_chunks += 1
                    self.audio_buffer.append(chunk)
                    
                    if self.silence_chunks > 25: # 800ms silence
                        self.is_speaking = False
                        self.deps.movement_manager.set_listening(False)
                        
                        if self.speech_chunks > 10:
                            full_audio = np.concatenate(self.audio_buffer)
                            asyncio.create_task(self._process_audio_input(full_audio))
                        
                        self.audio_buffer = []
                        self.speech_chunks = 0

    async def _process_audio_input(self, audio: np.ndarray):
        """Transcribe and run agent."""
        transcript = await asyncio.to_thread(self.stt.transcribe, audio)
        if not transcript.strip():
            return

        logger.info(f"Audio User: {transcript}")
        
        # Feedback to UI/Console
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
        
        prompt = f"[SOURCE: ROOM] User said: {transcript}"
        await self._run_agent(prompt)

    async def _run_agent(self, prompt: str):
        """Run the smolagents brain."""
        logger.info(f"Brain thinking on: {prompt}")
        
        try:
            # We run the agent in a thread to avoid blocking the asyncio loop
            # because smolagents might block while thinking/calling tools
            # Note: The tools we defined handle thread-safety by scheduling back to main loop
            response = await asyncio.to_thread(self.agent.run, prompt)
            
            logger.info(f"Agent finished: {response}")
            
            # If the agent returned a string but didn't use a tool (fallback), 
            # we should probably speak it if the source was ROOM, or text if SIGNAL.
            # But for now, we rely on the agent to use tools.
            # If response is string and tools weren't used? 
            # smolagents usually returns the final answer.
            
            # Simple heuristic: if tools were used, response might be "Done" or result.
            # If no tools used, response is the chat answer.
            # We should default to speaking if no explicit tool was used?
            # Or just log it.
            
            if isinstance(response, str) and len(response) > 0:
                 await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": response}))
                 
        except Exception as e:
            logger.error(f"Agent failed: {e}")

    async def emit(self):
        return await wait_for_item(self.output_queue)

    async def shutdown(self):
        if self.polling_task:
            self.polling_task.cancel()
