import asyncio
import base64

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from openai import AsyncOpenAI


class MinimalOpenaiRealtimeHandler(AsyncStreamHandler):
    def __init__(self, head_wobbler):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24000,
            input_sample_rate=24000,
        )
        self.head_wobbler = head_wobbler

        self.connection = None
        self.output_queue = asyncio.Queue()

    def copy(self):
        return MinimalOpenaiRealtimeHandler(self.head_wobbler)

    async def start_up(self):
        self.client = AsyncOpenAI()
        async with self.client.beta.realtime.connect(model="gpt-realtime") as conn:
            await conn.session.update(
                session={
                    "turn_detection": {
                        "type": "server_vad",
                    },
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "en",
                    },
                    "voice": "ballad",
                    "instructions": "You are a helpful assistant.",
                }
            )

            # Manage event received from the openai server
            self.connection = conn
            async for event in self.connection:
                # Handle interruptions
                if event.type == "input_audio_buffer.speech_started":
                    self.clear_queue()
                if (
                    event.type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "user", "content": event.transcript})
                    )
                if event.type == "response.audio_transcript.done":
                    await self.output_queue.put(
                        AdditionalOutputs(
                            {"role": "assistant", "content": event.transcript}
                        )
                    )
                if event.type == "response.audio.delta":
                    self.head_wobbler.feed(event.delta)
                    await self.output_queue.put(
                        (
                            self.output_sample_rate,
                            np.frombuffer(
                                base64.b64decode(event.delta), dtype=np.int16
                            ).reshape(1, -1),
                        ),
                    )

    # Microphone receive
    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.connection:
            return
        _, array = frame
        array = array.squeeze()
        audio_message = base64.b64encode(array.tobytes()).decode("utf-8")
        # Fills the input audio buffer to be sent to the server
        await self.connection.input_audio_buffer.append(audio=audio_message)  # type: ignore

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        # sends to the stream the stuff put in the output queue by the openai event handler
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        if self.connection:
            await self.connection.close()
            self.connection = None
