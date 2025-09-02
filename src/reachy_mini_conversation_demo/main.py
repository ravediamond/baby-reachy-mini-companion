from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
import warnings
import threading

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from websockets import ConnectionClosedError, ConnectionClosedOK

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from reachy_mini_conversation_demo.head_tracker import HeadTracker
from reachy_mini_conversation_demo.prompts import SESSION_INSTRUCTIONS
from reachy_mini_conversation_demo.tools import (
    Deps,
    TOOL_SPECS,
    dispatch_tool_call,
)
from reachy_mini_conversation_demo.audio_sway import AudioSync, AudioConfig, pcm_to_b64
from reachy_mini_conversation_demo.movement import MovementManager
from reachy_mini_conversation_demo.gstreamer import GstPlayer, GstRecorder
from reachy_mini_conversation_demo.vision import VisionManager, init_vision, init_camera

# env + logging
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress WebRTC warnings
warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

# Reduce logging noise
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("fastrtc").setLevel(logging.ERROR)
logging.getLogger("aioice").setLevel(logging.WARNING)


# Read from .env
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))
SIM = os.getenv("SIM", "false").lower() in ("true", "1", "yes", "on")
VISION_ENABLED = os.getenv("VISION_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
    "on",
)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-realtime-preview")

HEAD_TRACKING = os.getenv("HEAD_TRACKING", "false").lower() in (
    "true",
    "1",
    "yes",
    "on",
)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logger.error("OPENAI_API_KEY not set! Please add it to your .env file.")
    raise RuntimeError("OPENAI_API_KEY missing")
masked = (API_KEY[:6] + "..." + API_KEY[-4:]) if len(API_KEY) >= 12 else "<short>"
logger.info("OPENAI_API_KEY loaded (prefix): %s", masked)

# init camera
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
camera = init_camera(camera_index=CAMERA_INDEX, simulation=SIM)

# Vision manager initialization with proper error handling
vision_manager: VisionManager | None = None

if camera and camera.isOpened() and VISION_ENABLED:
    vision_manager = init_vision(camera=camera)

# hardware / IO
current_robot = ReachyMini()

head_tracker: HeadTracker = None

if HEAD_TRACKING and not SIM:
    head_tracker = HeadTracker()
    logger.info("Head tracking enabled")
elif HEAD_TRACKING and SIM:
    logger.warning("Head tracking disabled while in Simulation")
else:
    logger.warning("Head tracking disabled")

movement_manager = MovementManager(
    current_robot=current_robot, head_tracker=head_tracker, camera=camera
)
robot_is_speaking = asyncio.Event()
speaking_queue = asyncio.Queue()


# tool deps
deps = Deps(
    reachy_mini=current_robot,
    create_head_pose=create_head_pose,
    camera=camera,
    vision_manager=vision_manager,
)

# audio sync
audio_sync = AudioSync(
    AudioConfig(output_sample_rate=SAMPLE_RATE),
    set_offsets=movement_manager.set_offsets,
)


class OpenAIRealtimeHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            input_sample_rate=SAMPLE_RATE,
        )
        self.client: AsyncOpenAI | None = None
        self.connection = None
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self._stop = False
        self._started_audio = False
        self._connection_ready = False
        self._speech_start_time = 0.0
        # backoff managment for retry
        self._backoff_start = 1.0
        self._backoff_max = 16.0
        self._backoff = self._backoff_start

    def copy(self):
        return OpenAIRealtimeHandler()

    async def start_up(self):
        if not self._started_audio:
            audio_sync.start()
            self._started_audio = True

        if self.client is None:
            logger.info("Realtime start_up: creating AsyncOpenAI client...")
            self.client = AsyncOpenAI(api_key=API_KEY)

        self._backoff = self._backoff_start
        while not self._stop:
            try:
                async with self.client.beta.realtime.connect(
                    model=MODEL_NAME
                ) as rt_connection:
                    self.connection = rt_connection
                    self._connection_ready = False

                    # configure session
                    await rt_connection.session.update(
                        session={
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.6,  # Higher threshold = less sensitive
                                "prefix_padding_ms": 300,  # More padding before speech
                                "silence_duration_ms": 800,  # Longer silence before detecting end
                            },
                            "voice": "ballad",
                            "instructions": SESSION_INSTRUCTIONS,
                            "input_audio_transcription": {
                                "model": "whisper-1",
                                "language": "en",
                            },
                            "tools": TOOL_SPECS,
                            "tool_choice": "auto",
                            "temperature": 0.7,
                        }
                    )

                    # Wait for session to be configured
                    await asyncio.sleep(0.2)

                    # Add system message with even stronger brevity emphasis
                    await rt_connection.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"{SESSION_INSTRUCTIONS}\n\nIMPORTANT: Always keep responses under 25 words. Be extremely concise.",
                                }
                            ],
                        }
                    )

                    self._connection_ready = True

                    logger.info(
                        "Session updated: tools=%d, voice=%s, vad=improved",
                        len(TOOL_SPECS),
                        "ballad",
                    )

                    logger.info("Realtime event loop started with improved VAD")
                    self._backoff = self._backoff_start

                    async for event in rt_connection:
                        event_type = getattr(event, "type", None)
                        logger.debug("RT event: %s", event_type)

                        # Enhanced speech state tracking
                        if event_type == "input_audio_buffer.speech_started":
                            # Only process user speech if robot isn't currently speaking
                            if not robot_is_speaking.is_set():
                                audio_sync.on_input_speech_started()
                                logger.info("User speech detected (robot not speaking)")
                            else:
                                logger.info(
                                    "Ignoring speech detection - robot is speaking"
                                )

                        elif event_type == "response.started":
                            self._speech_start_time = time.time()
                            audio_sync.on_response_started()
                            logger.info("Robot started speaking")

                        elif event_type in (
                            "response.audio.completed",
                            "response.completed",
                            "response.audio.done",
                        ):
                            logger.info("Robot finished speaking %s", event_type)

                        elif (
                            event_type
                            == "conversation.item.input_audio_transcription.completed"
                        ):
                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {"role": "user", "content": event.transcript}
                                )
                            )

                        elif event_type == "response.audio_transcript.done":
                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {"role": "assistant", "content": event.transcript}
                                )
                            )

                        # audio streaming
                        if event_type == "response.audio.delta":
                            robot_is_speaking.set()
                            # block mic from recording for given time, for each audio delta
                            speaking_queue.put_nowait(0.25)
                            audio_sync.on_response_audio_delta(
                                getattr(event, "delta", b"")
                            )

                        elif event_type == "response.function_call_arguments.done":
                            tool_name = getattr(event, "name", None)
                            args_json_str = getattr(event, "arguments", None)
                            call_id = getattr(event, "call_id", None)

                            try:
                                tool_result = await dispatch_tool_call(
                                    tool_name, args_json_str, deps
                                )
                            except Exception as e:
                                logger.exception("Tool %s failed", tool_name)
                                tool_result = {"error": str(e)}

                            await rt_connection.conversation.item.create(
                                item={
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": json.dumps(tool_result),
                                }
                            )
                            logger.info(
                                "Sent tool=%s call_id=%s result=%s",
                                tool_name,
                                call_id,
                                tool_result,
                            )
                            if tool_name and (
                                tool_name == "camera" or "scene" in tool_name
                            ):
                                logger.info(
                                    "Forcing response after tool call %s", tool_name
                                )
                                await rt_connection.response.create()

                        # server errors
                        if event_type == "error":
                            err = getattr(event, "error", None)
                            msg = getattr(
                                err, "message", str(err) if err else "unknown error"
                            )
                            logger.error("Realtime error: %s (raw=%s)", msg, err)
                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {"role": "assistant", "content": f"[error] {msg}"}
                                )
                            )

            except (ConnectionClosedOK, ConnectionClosedError) as e:
                if self._stop:
                    break
                logger.warning(
                    "Connection closed (%s). Reconnecting…",
                    getattr(e, "code", "no-code"),
                )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Realtime loop error; will reconnect")
            finally:
                self.connection = None
                self._connection_ready = False

            # Exponential backoff
            delay = min(self._backoff, self._backoff_max) + random.uniform(0, 0.5)
            logger.info("Reconnect in %.1fs…", delay)
            await asyncio.sleep(delay)
            self._backoff = min(self._backoff * 2.0, self._backoff_max)

    async def receive(self, frame: bytes) -> None:
        """Mic frames from fastrtc."""
        # Don't send mic audio while robot is speaking (simple echo cancellation)
        if robot_is_speaking.is_set() or not self._connection_ready:
            return

        mic_samples = np.frombuffer(frame, dtype=np.int16).squeeze()
        audio_b64 = pcm_to_b64(mic_samples)

        try:
            await self.connection.input_audio_buffer.append(audio=audio_b64)
        except (ConnectionClosedOK, ConnectionClosedError):
            pass

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        """Return audio for playback or chat outputs."""
        try:
            sample_rate, pcm_frame = audio_sync.playback_q.get_nowait()
            logger.debug(
                "Emitting playback frame (sr=%d, n=%d)", sample_rate, pcm_frame.size
            )
            return (sample_rate, pcm_frame)
        except asyncio.QueueEmpty:
            pass
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        logger.info("Shutdown: closing connections and audio")
        self._stop = True
        if self.connection:
            try:
                await self.connection.close()
            except Exception:
                logger.exception("Error closing realtime connection")
            finally:
                self.connection = None
                self._connection_ready = False
        await audio_sync.stop()


async def receive_loop(recorder: GstRecorder, openai: OpenAIRealtimeHandler) -> None:
    logger.info("Starting receive loop")
    while not stop_event.is_set():
        data = recorder.get_sample()
        if data is not None:
            await openai.receive(data)
        await asyncio.sleep(0)  # Prevent busy waiting


async def emit_loop(player: GstPlayer, openai: OpenAIRealtimeHandler) -> None:
    while not stop_event.is_set():
        data = await openai.emit()
        if isinstance(data, AdditionalOutputs):
            for msg in data.args:
                content = msg.get("content", "")
                logger.info(
                    "role=%s content=%s",
                    msg.get("role"),
                    content if len(content) < 500 else content[:500] + "…",
                )

        elif isinstance(data, tuple):
            _, frame = data
            player.push_sample(frame.tobytes())

        else:
            pass
        await asyncio.sleep(0)  # Prevent busy waiting


async def control_mic_loop():
    # Control mic to prevent echo, blocks mic for given time
    while not stop_event.is_set():
        try:
            block_time = speaking_queue.get_nowait()
        except asyncio.QueueEmpty:
            robot_is_speaking.clear()
            audio_sync.on_response_completed()
            await asyncio.sleep(0)
            continue

        await asyncio.sleep(block_time)


stop_event = threading.Event()


async def main():
    openai = OpenAIRealtimeHandler()
    recorder = GstRecorder()
    recorder.record()
    player = GstPlayer()
    player.play()

    movement_manager.set_neutral()
    logger.info("Starting main audio loop. You can start to speak")

    tasks = [
        asyncio.create_task(openai.start_up(), name="openai"),
        asyncio.create_task(emit_loop(player, openai), name="emit"),
        asyncio.create_task(receive_loop(recorder, openai), name="recv"),
        asyncio.create_task(control_mic_loop(), name="mic-mute"),
        asyncio.create_task(
            movement_manager.enable(stop_event=stop_event), name="move"
        ),
    ]

    if vision_manager:
        tasks.append(
            asyncio.create_task(
                vision_manager.enable(stop_event=stop_event), name="vision"
            ),
        )

    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    except asyncio.CancelledError:
        logger.info("Shutting down")
        stop_event.set()

    if camera:
        camera.release()

    await openai.shutdown()
    movement_manager.set_neutral()
    recorder.stop()
    player.stop()

    current_robot.client.disconnect()
    logger.info("Stopped, robot disconected")


if __name__ == "__main__":
    asyncio.run(main())
