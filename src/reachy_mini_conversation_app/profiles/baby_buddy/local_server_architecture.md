# Local AI Server Architecture (Jetson Orin NX)

This document outlines the architecture for offloading the "Brain" (LLM), "Ears" (STT), and "Mouth" (TTS) to a powerful local server (Jetson Orin NX), keeping the robot client lightweight.

## Concept

Instead of the robot running heavy AI models or connecting to OpenAI cloud, it connects to a local **Jetson Orin NX** over the LAN via WebSockets.

The architecture mimics the "Audio-in / Audio-out" flow of the OpenAI Realtime API. This allows the robot to remain a "thin client" that only handles:
1.  **Audio Capture** (Microphone)
2.  **Audio Playback** (Speakers)
3.  **Motor Control** (Head movement, Antennas)
4.  **Vision Capture** (Camera frames)

## 1. The Jetson Server (The "Brain")

This is a Python service (FastAPI + WebSockets) running on the Jetson.

### Hardware Requirements
- **Device**: Jetson Orin NX (16GB RAM)
- **OS**: JetPack 6.x (Ubuntu 22.04)

### Software Stack
| Component | Technology | Implementation |
| :--- | :--- | :--- |
| **VAD** (Voice Activity Detection) | **Silero VAD** | Very low latency, runs on CPU/GPU. Detects when user stops speaking. |
| **STT** (Speech-to-Text) | **Faster-Whisper** | `small.en` or `medium.en` model running on GPU (CUDA). |
| **LLM** (Intelligence) | **Llama.cpp** | `Llama-3-8B-Instruct` (GGUF, 4-bit/8-bit quant) with GPU offload (`n_gpu_layers`). |
| **TTS** (Text-to-Speech) | **Piper** or **StyleTTS2** | Fast, high-quality offline TTS. |
| **Server** | **FastAPI** | Handles WebSocket connections. |

### Data Flow (Pipeline)
1.  **Receive**: Server receives raw PCM audio chunks from Robot via WebSocket.
2.  **VAD**: Accumulates audio. If silence is detected > 500ms, triggers processing.
3.  **Transcribe**: Audio buffer -> Faster-Whisper -> User Text.
4.  **Think**: User Text -> Llama.cpp -> Assistant Text (streamed tokens).
5.  **Synthesize**: Assistant Text Stream -> TTS -> Audio Bytes.
6.  **Send**: Audio Bytes -> WebSocket -> Robot.

## 2. The Robot Client (This App)

The robot application needs a new **`LocalSocketHandler`** class that mirrors the `OpenaiRealtimeHandler` but speaks a simpler protocol.

### Changes Required
1.  **New Handler**: `src/reachy_mini_conversation_app/local_socket_handler.py`
    *   Connects to `ws://<JETSON_IP>:<PORT>/ws`.
    *   Sends microphone audio.
    *   Receives audio chunks (to play) + text (to display).
    *   Handles "Tool Calls" (if the local LLM outputs a special JSON format).

2.  **CLI Arguments**:
    *   `--local-server-ip`: IP of the Jetson.
    *   `--local-server-port`: Port (e.g., 8000).

### Protocol Example (Simplified JSON)

**Robot -> Server:**
```json
{ "type": "audio", "data": "<base64_pcm_chunk>" }
```

**Server -> Robot:**
```json
{ "type": "transcript_user", "text": "Hello robot." }
{ "type": "audio_delta", "data": "<base64_pcm_chunk>" }
{ "type": "transcript_assistant", "text": "Hi there!" }
{ "type": "tool_call", "name": "dance", "args": "{}" }
```

## 3. Advantages
1.  **Privacy**: No data leaves the local network.
2.  **Latency**: Local LAN speed + GPU acceleration (often faster than cloud round-trip).
3.  **Cost**: Zero API fees.
4.  **Modularity**: You can swap the LLM (e.g., from Llama 3 to Mistral) on the Jetson without touching the robot code.

## 4. Implementation Plan

1.  **Build the Jetson Server**: Create a separate repo (e.g., `reachy-mini-brain-server`) with the VAD/STT/LLM/TTS pipeline.
2.  **Update Robot App**: Add the `LocalSocketHandler` to this repository to connect to it.
