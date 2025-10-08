import asyncio
import base64
import logging
import numpy as np

from fastrtc import AdditionalOutputs
from gi.repository import Gst

from reachy_mini_conversation_demo.audio.head_wobbler import SAMPLE_RATE
from reachy_mini_conversation_demo.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_demo.audio.gstreamer import GstPlayer, GstRecorder

logger = logging.getLogger(__name__)


class LocalStream:
    def __init__(self, handler: OpenaiRealtimeHandler):
        self.handler = handler
        self._stop_event = asyncio.Event()

        self.recorder = GstRecorder(sample_rate=SAMPLE_RATE)
        self.player = GstPlayer(sample_rate=SAMPLE_RATE)

        self.handler._clear_queue = self.clear_queue  # type: ignore[assignment]

    #     player_bus = self.player.pipeline.get_bus()
    #     player_bus.add_signal_watch()
    #     player_bus.connect("message", self.on_player_message)

    # def on_player_message(self, bus, message):
    #     # logger.info(f"Player message: {message.type}")
    #     if message.type == Gst.MessageType.STATE_CHANGED:
    #         old_state, new_state, pending_state = message.parse_state_changed()
    #         if new_state != old_state and new_state == Gst.State.PLAYING:
    #             print("Player is now playing")
    #             self.recorder.pipeline.set_state(Gst.State.PAUSED)

    #         if new_state != old_state and new_state == Gst.State.PAUSED:
    #             print("Player is now paused")
    #             self.recorder.pipeline.set_state(Gst.State.PLAYING)

    #     if message.type == Gst.MessageType.EOS:
    #         self.recorder.pipeline.set_state(Gst.State.PLAYING)
    #         print("Player reached end of stream, restarting recorder")

    def clear_queue(self):
        self.player.pipeline.set_state(Gst.State.PAUSED)
        self.player.appsrc.send_event(Gst.Event.new_flush_start())
        self.player.appsrc.send_event(Gst.Event.new_flush_stop(reset_time=True))
        self.player.pipeline.set_state(Gst.State.PLAYING)
        logger.info("Cleared player queue")

    def start(self):
        self._stop_event.clear()
        self.recorder.record()
        self.player.play()

        async def runner():
            tasks = [
                asyncio.create_task(self.handler.start_up(), name="openai-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            await asyncio.gather(*tasks)

        asyncio.run(runner())

    def stop(self):
        self._stop_event.set()
        self.recorder.stop()
        self.player.stop()

    async def record_loop(self) -> None:
        logger.info("Starting receive loop")
        while not self._stop_event.is_set():
            data = self.recorder.get_sample()

            if data is not None:
                frame = np.frombuffer(data, dtype=np.int16).squeeze()
                await self.handler.receive((0, frame))
            await asyncio.sleep(0)  # Prevent busy waiting

    async def play_loop(self) -> None:
        while not self._stop_event.is_set():
            data = await self.handler.emit()
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
                self.player.push_sample(frame.tobytes())
            else:
                pass

            await asyncio.sleep(0)  # Prevent busy waiting
