from __future__ import annotations
import re
import json
import asyncio
import logging
from typing import TYPE_CHECKING, Tuple, Optional
from collections import deque

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.local.llm import LocalLLM
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_tool_specs, dispatch_tool_call
from reachy_mini_conversation_app.input.signal_interface import SignalInterface


if TYPE_CHECKING:
    from reachy_mini_conversation_app.local.stt import LocalSTT
    from reachy_mini_conversation_app.local.tts import LocalTTS
    from reachy_mini_conversation_app.local.vad import SileroVAD
    from reachy_mini_conversation_app.audio.classifier import AudioClassifier


logger = logging.getLogger(__name__)


class LocalSessionHandler(AsyncStreamHandler):
    """Local processing handler for Reachy Mini (VAD -> STT -> LLM -> TTS + Signal)."""

    def __init__(
        self,
        deps: ToolDependencies,
        llm_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        enable_signal: bool = True,
    ):
        """Initialize the local session handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24000,
            input_sample_rate=16000,
        )
        self.deps = deps
        # Inject speech callback into dependencies so tools like 'speak' can use it
        self.deps.speak_func = self._process_sentence

        self.llm_url = llm_url or config.LOCAL_LLM_URL
        self.llm_model = llm_model or config.LOCAL_LLM_MODEL
        self.enable_signal = enable_signal

        self.output_queue: asyncio.Queue[object] = asyncio.Queue()
        self.audio_buffer: list[np.ndarray] = []
        self.lookback_buffer: deque[np.ndarray] = deque(maxlen=20)  # 20 * 32ms = 640ms lookback
        self.vad_buffer = np.array([], dtype=np.float32)

        # Audio Classification Buffer (1s window)
        self.classifier_buffer = np.array([], dtype=np.float32)
        self.classifier: Optional[AudioClassifier] = None
        self.last_cry_time: float = 0.0

        # Vision Danger Detection
        self.danger_detector = None
        self.danger_detection_task: Optional[asyncio.Task[None]] = None
        self.last_danger_time: float = 0.0

        self.is_speaking = False
        self.silence_chunks = 0
        self.speech_chunks = 0

        # Initialize models lazily or in start_up
        self.vad: Optional[SileroVAD] = None
        self.stt: Optional[LocalSTT] = None
        self.llm: Optional[LocalLLM] = None
        self.tts: Optional[LocalTTS] = None
        self.pipeline_task: Optional[asyncio.Task[None]] = None

        # Signal integration
        self.signal: Optional[SignalInterface] = None
        self.signal_polling_task: Optional[asyncio.Task[None]] = None
        self.user_phone = config.SIGNAL_USER_PHONE

        # Tool specs cache (rebuilt in start_up with feature-based exclusions)
        self.tool_specs = get_tool_specs()

    def copy(self) -> "LocalSessionHandler":
        """Create a copy of the handler."""
        return LocalSessionHandler(self.deps, self.llm_url, self.llm_model, self.enable_signal)

    def _build_tool_exclusions(self) -> list[str]:
        """Build tool exclusion list based on feature flags."""
        exclusions: list[str] = []
        if not config.FEATURE_AUTO_SOOTHE:
            exclusions.extend(["soothe_baby", "check_baby_crying"])
        if not config.FEATURE_STORY_TIME:
            exclusions.append("story_time")
        if not config.FEATURE_SIGNAL_ALERTS:
            exclusions.extend(["send_signal", "send_signal_photo"])
        if not config.FEATURE_DANGER_DETECTION:
            exclusions.append("check_danger")
        return exclusions

    async def start_up(self):
        """Initialize local models."""
        logger.info("Initializing Local AI Pipeline...")

        # Lazy imports: these pull in heavy deps (torch, faster-whisper, kokoro-onnx, onnxruntime)
        # at import time, so we defer to avoid slowing down module loading.
        from reachy_mini_conversation_app.local.stt import LocalSTT
        from reachy_mini_conversation_app.local.tts import LocalTTS
        from reachy_mini_conversation_app.local.vad import SileroVAD

        try:
            # Rebuild tool specs with feature-based exclusions
            exclusions = self._build_tool_exclusions()
            if exclusions:
                logger.info(f"Feature flags: excluding tools {exclusions}")
            self.tool_specs = get_tool_specs(exclusion_list=exclusions)

            logger.info("Loading VAD (Silero)...")
            self.vad = await asyncio.to_thread(SileroVAD)

            logger.info(f"Loading STT (Faster-Whisper {config.LOCAL_STT_MODEL})...")
            self.stt = await asyncio.to_thread(LocalSTT, model_size=config.LOCAL_STT_MODEL)

            logger.info("Loading LLM Client...")
            self.llm = LocalLLM(base_url=self.llm_url, model=self.llm_model)

            logger.info("Loading TTS (Kokoro)...")
            self.tts = await asyncio.to_thread(LocalTTS)

            # Audio Classifier (YAMNet) — gated by feature flag
            if config.FEATURE_CRY_DETECTION:
                logger.info("Loading Audio Classifier (YAMNet)...")
                try:
                    from reachy_mini_conversation_app.audio.classifier import AudioClassifier

                    self.classifier = await asyncio.to_thread(AudioClassifier)
                except ImportError:
                    logger.info("onnxruntime not installed, audio classification disabled.")
                except Exception as e:
                    logger.warning(f"Audio Classifier failed to load: {e}")
            else:
                logger.info("Baby cry detection disabled by feature flag.")

            # Load system prompt
            logger.info("Setting system prompt...")
            self.llm.set_system_prompt(get_session_instructions())

            # Pre-warm LLM with tools to avoid cold start delay
            logger.info("Pre-warming LLM with tools...")
            try:
                async for _ in self.llm.chat_stream(user_text="hi", tools=self.tool_specs):
                    pass
                # Clear history after warmup
                self.llm.history = []
                logger.info("LLM pre-warm complete.")
            except Exception as e:
                logger.warning(f"LLM pre-warm failed (non-critical): {e}")

            # Danger Detector (YOLO) — gated by feature flag
            if config.FEATURE_DANGER_DETECTION and self.deps.camera_worker is not None:
                logger.info("Loading Danger Detector (YOLO)...")
                try:
                    from reachy_mini_conversation_app.vision.danger_detector import DangerDetector

                    self.danger_detector = await asyncio.to_thread(DangerDetector)  # type: ignore[func-returns-value,arg-type]
                    self.danger_detection_task = asyncio.create_task(self._poll_danger_detection())
                    logger.info("Danger detection started.")
                except ImportError:
                    logger.info("YOLO not installed, danger detection disabled.")
                except Exception as e:
                    logger.warning(f"Danger Detector failed to load: {e}")
            elif not config.FEATURE_DANGER_DETECTION:
                logger.info("Danger detection disabled by feature flag.")

            # Head Tracking — gated by feature flag
            if config.FEATURE_HEAD_TRACKING and self.deps.camera_worker is not None:
                if self.deps.camera_worker.head_tracker is None:
                    logger.info("Loading Head Tracker (MediaPipe)...")
                    try:
                        from reachy_mini_toolbox.vision import HeadTracker

                        tracker = await asyncio.to_thread(HeadTracker)
                        self.deps.camera_worker.head_tracker = tracker
                        self.deps.camera_worker.set_head_tracking_enabled(True)
                        logger.info("Head tracking (MediaPipe) enabled.")
                    except ImportError:
                        logger.info("MediaPipe not installed, head tracking disabled.")
                    except Exception as e:
                        logger.warning(f"Head Tracker failed to load: {e}")
            elif not config.FEATURE_HEAD_TRACKING and self.deps.camera_worker is not None:
                self.deps.camera_worker.set_head_tracking_enabled(False)
                logger.info("Head tracking disabled by feature flag.")

            # Signal — gated by feature flag
            if self.enable_signal and config.FEATURE_SIGNAL_ALERTS:
                self.signal = SignalInterface()
                if self.signal.available:
                    self.signal_polling_task = asyncio.create_task(self._poll_signal())
                    logger.info("Signal polling started.")
                else:
                    logger.info("Signal not available, skipping.")
            elif not config.FEATURE_SIGNAL_ALERTS:
                logger.info("Signal alerts disabled by feature flag.")

            logger.info("Local Pipeline Ready.")
        except Exception as e:
            logger.error(f"Failed to initialize local pipeline: {e}")
            logger.error("Pipeline will remain inactive. The settings dashboard is still accessible.")

    async def receive(self, frame: Tuple[int, np.ndarray]) -> None:
        """Process incoming audio frame."""
        sr, audio = frame

        # 1. Convert to float32 safely
        if audio.dtype == np.float32:
            audio_float = audio.copy()
        else:
            audio_float = audio.astype(np.float32) / 32768.0

        # Apply microphone gain
        if config.MIC_GAIN != 1.0:
            audio_float = audio_float * config.MIC_GAIN

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

        # 4.5 Add to Classifier Buffer
        if self.classifier:
            self.classifier_buffer = np.concatenate((self.classifier_buffer, audio_float))
            # Process every ~1 second (16000 samples)
            if len(self.classifier_buffer) >= 16000:
                chunk_to_classify = self.classifier_buffer[:16000]
                self.classifier_buffer = self.classifier_buffer[16000:]  # Slide window

                # Check for cry (throttled to once every 10s to avoid spam)
                import time

                now = time.time()
                if now - self.last_cry_time > 10.0:
                    # Run in thread
                    results = await asyncio.to_thread(self.classifier.classify, chunk_to_classify)
                    for label, score in results:
                        if label in ["Baby cry, infant cry", "Crying, sobbing", "Whimper"] and score > 0.4:
                            logger.info(f"Audio Event Detected: {label} ({score:.2f})")
                            self.last_cry_time = now

                            # Update shared status for tools
                            if self.deps.audio_classifier_status is not None:
                                self.deps.audio_classifier_status["latest_event"] = label
                                self.deps.audio_classifier_status["timestamp"] = now
                                self.deps.audio_classifier_status["score"] = float(score)

                            # Inject event into LLM context
                            asyncio.create_task(self._process_system_event(f"I hear a {label.lower()} nearby."))
                            break

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

                    # Prepend lookback buffer to start of speech
                    self.audio_buffer = list(self.lookback_buffer)
                    self.lookback_buffer.clear()

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

                    # Silence threshold: 1.5s ~ 47 chunks (47 * 32ms = 1504ms)
                    if self.silence_chunks > 47:
                        self.is_speaking = False
                        logger.info(f"Speech finished ({self.speech_chunks} chunks)")
                        self.deps.movement_manager.set_listening(False)

                        # Trigger pipeline if we had enough speech
                        if self.speech_chunks > 5:  # Lowered from 10 to catch short commands
                            full_audio = np.concatenate(self.audio_buffer)
                            logger.info("Triggering AI pipeline...")
                            asyncio.create_task(self._run_pipeline(full_audio))
                        else:
                            logger.debug("Speech too short, ignoring.")

                        self.audio_buffer = []
                        self.speech_chunks = 0
                else:
                    # Not speaking, keep filling lookback buffer
                    self.lookback_buffer.append(chunk)

    async def _process_system_event(self, event_text: str):
        """Process a system event (like 'Baby cry detected') as if it were a user prompt."""
        if self.llm is None:
            return
        logger.info(f"System Event: {event_text}")
        await self.output_queue.put(AdditionalOutputs({"role": "system", "content": event_text}))

        # 2. LLM Loop
        current_input = f"[System Notification: {event_text}]"

        # Reuse the logic from _run_pipeline essentially, but simplified
        max_turns = 3
        turn = 0
        tool_outputs = None

        while turn < max_turns:
            turn += 1
            full_response_text = ""
            current_sentence = ""

            llm_user_input = current_input if turn == 1 else None

            logger.info(f"LLM Turn {turn} (Event: {llm_user_input or 'Tool Outputs'})")

            tool_calls = []

            async for event in self.llm.chat_stream(
                user_text=llm_user_input, tools=self.tool_specs, tool_outputs=tool_outputs
            ):
                if event["type"] == "text":
                    token = event["content"]
                    full_response_text += token
                    current_sentence += token
                    if token in [".", "!", "?", "\n"]:
                        await self._process_sentence(current_sentence)
                        current_sentence = ""
                elif event["type"] == "tool_call":
                    tool_calls.append(event["tool_call"])
                elif event["type"] == "error":
                    logger.error(f"LLM Error: {event['content']}")

            if current_sentence.strip():
                await self._process_sentence(current_sentence)

            if full_response_text:
                logger.info(f"Assistant: {full_response_text}")
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": full_response_text}))

            if not tool_calls:
                break

            tool_outputs = []
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                args_str = tc["function"]["arguments"]
                call_id = tc["id"]
                logger.info(f"🛠️ Tool Call: {func_name}({args_str})")
                result = await dispatch_tool_call(func_name, args_str, self.deps)
                result_str = json.dumps(result)
                tool_outputs.append({"role": "tool", "content": result_str, "tool_call_id": call_id})
                logger.info(f"   -> Result: {result_str}")

    async def _run_pipeline(self, audio: np.ndarray):
        """Run STT -> LLM -> TTS pipeline."""
        if self.stt is None or self.llm is None or self.tts is None:
            return
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

        tool_outputs = None  # For subsequent turns

        while turn < max_turns:
            turn += 1
            full_response_text = ""
            current_sentence = ""

            # If this is the first turn, we pass user input. Subsequent turns are tool outputs.
            llm_user_input = current_input if turn == 1 else None

            logger.info(f"LLM Turn {turn} (Input: {llm_user_input or 'Tool Outputs'})")

            tool_calls = []

            async for event in self.llm.chat_stream(
                user_text=llm_user_input, tools=self.tool_specs, tool_outputs=tool_outputs
            ):
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

                tool_outputs.append({"role": "tool", "content": result_str, "tool_call_id": call_id})

                logger.info(f"   -> Result: {result_str}")

            # Prepare for next turn
            # We Loop back with tool_outputs, llm_user_input will be None

    async def _process_sentence(self, text: str):
        """Synthesize and queue audio for a sentence, handling tone and embedded commands."""
        if self.tts is None:
            return
        # --- 1. Extract and Execute Commands (Fallback / Legacy) ---
        # Look for PLAY_EMOTION("emotion_name") - Kept for backward compatibility or if LLM hallucinates text
        emotion_matches = re.findall(r'play_emotion\s*\(\s*["\']([^"\']+)["\']\s*\)', text, re.IGNORECASE)
        for emotion in emotion_matches:
            logger.info(f"⚡️ Executing implied command (Legacy): play_emotion('{emotion}')")
            asyncio.create_task(dispatch_tool_call("play_emotion", f'{{"emotion_name": "{emotion}"}}', self.deps))

        # --- 2. Clean Text for TTS ---
        # Remove the command strings we just found (more robust regex)
        text = re.sub(r"\(?play_emotion\s*\([^)]+\)\)?", "", text, flags=re.IGNORECASE)

        # Remove bold/italic markdown (*word*, **word**)
        text = text.replace("**", "").replace("*", "")

        # Remove action descriptions in parentheses or asterisks if they remain
        # e.g. (waves hands) or *laughs*
        text = re.sub(r"\s*\([^)]*\)", "", text)

        # Remove emojis (Broad unicode range for common emojis)
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text)

        # Remove other common artifacts if any
        text = text.strip()

        if not text:
            return

        # Simple Tone Analysis
        speed = 1.0
        voice = "af_sarah"  # Default cheerful

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

            b64_data = base64.b64encode(audio_int16.tobytes()).decode("utf-8")
            self.deps.head_wobbler.feed(b64_data)

    async def _poll_danger_detection(self):
        """Background task: run YOLO on camera frames to detect dangerous objects."""
        import time

        if self.danger_detector is None or self.deps.camera_worker is None:
            return
        logger.info("Starting Danger Detection Poller...")
        while True:
            try:
                frame = self.deps.camera_worker.get_latest_frame()
                if frame is not None:
                    detections = await asyncio.to_thread(self.danger_detector.detect, frame)

                    if detections:
                        now = time.time()
                        # Throttle: one alert per 30 seconds
                        if now - self.last_danger_time > 30.0:
                            self.last_danger_time = now

                            # Update shared status for tools
                            if self.deps.vision_threat_status is not None:
                                self.deps.vision_threat_status["latest_threat"] = detections[0]["label"]
                                self.deps.vision_threat_status["timestamp"] = now
                                self.deps.vision_threat_status["objects"] = detections

                            labels = ", ".join(d["label"] for d in detections)
                            logger.info(f"Danger Detected: {labels}")

                            # Inject system event — LLM decides what to do (camera analysis, Signal alert, etc.)
                            asyncio.create_task(
                                self._process_system_event(
                                    f"I see a potentially dangerous object near the baby: {labels}. "
                                    "Please take a photo with the camera tool for a closer look and alert the parent via Signal."
                                )
                            )

                await asyncio.sleep(2.0)  # Check every 2 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Danger detection error: {e}")
                await asyncio.sleep(5.0)

    async def _poll_signal(self):
        """Background task to poll Signal messages."""
        if self.signal is None:
            return
        logger.info("Starting Signal Poller...")
        while True:
            try:
                messages = await self.signal.poll_messages()
                for msg in messages:
                    sender = msg["sender"]
                    text = msg["content"]
                    logger.info(f"Signal received from {sender}: {text}")

                    # Process through LLM and respond via Signal
                    await self._handle_signal_message(sender, text)

                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Signal polling error: {e}")
                await asyncio.sleep(5.0)

    async def _handle_signal_message(self, sender: str, text: str):
        """Process a Signal message and respond, supporting tool use."""
        if self.llm is None:
            return
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": f"[Signal] {text}"}))

        current_input = text
        max_turns = 3
        turn = 0
        tool_outputs = None

        while turn < max_turns:
            turn += 1
            full_response_text = ""

            # First turn uses user text, subsequent turns use None (and rely on tool_outputs)
            llm_user_input = current_input if turn == 1 else None

            tool_calls = []

            async for event in self.llm.chat_stream(
                user_text=llm_user_input, tools=self.tool_specs, tool_outputs=tool_outputs
            ):
                if event["type"] == "text":
                    full_response_text += event["content"]
                elif event["type"] == "tool_call":
                    tool_calls.append(event["tool_call"])
                elif event["type"] == "error":
                    logger.error(f"LLM Error: {event['content']}")

            # If we have text response, log it (we'll send it at end of turn or if final)
            if full_response_text.strip():
                # If there are tool calls, this might be a "thought" or preamble.
                # If no tool calls, it's the final answer.
                pass

            # If no tool calls, we are done with this chain
            if not tool_calls:
                if full_response_text.strip():
                    logger.info(f"Signal Response: {full_response_text}")
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "assistant", "content": f"[Signal] {full_response_text}"})
                    )

                    target = sender if sender else self.user_phone
                    if target and self.signal:
                        await self.signal.send_message(full_response_text, target)
                break

            # Execute Tools
            tool_outputs = []
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                args_str = tc["function"]["arguments"]
                call_id = tc["id"]

                logger.info(f"🛠️ Signal Tool Call: {func_name}({args_str})")

                # Execute
                result = await dispatch_tool_call(func_name, args_str, self.deps)
                result_str = json.dumps(result)

                tool_outputs.append({"role": "tool", "content": result_str, "tool_call_id": call_id})

                logger.info(f"   -> Result: {result_str}")

            # Loop continues to next turn to let LLM react to tool outputs

    async def emit(self):
        """Return the next output from the handler queue."""
        return await wait_for_item(self.output_queue)

    async def shutdown(self):
        """Shutdown the handler and cancel background tasks."""
        if self.danger_detection_task:
            self.danger_detection_task.cancel()
        if self.signal_polling_task:
            self.signal_polling_task.cancel()
