# Local Mac Architecture (All-in-One)

This architecture allows the Reachy Mini companion to run **100% locally** on a Mac M1/M2/M3 (or powerful PC), offering high privacy, low latency, and zero API costs.

## Concept

We replace the OpenAI Realtime API (which handles Hearing, Thinking, and Speaking in the cloud) with a modular pipeline of specialized local models running on the host machine.

The robot remains a USB peripheral. The Mac acts as the complete brain and body controller.

## The Stack

| Component | Function | Model / Library | Hardware Usage |
| :--- | :--- | :--- | :--- |
| **VAD** | Voice Activity Detection | **Silero VAD** | CPU (Negligible) |
| **STT** | Hearing (Speech-to-Text) | **Faster-Whisper** (`small.en` or `base.en`) | CPU / Neural Engine |
| **LLM** | Thinking (Intelligence) | **Ollama** running **Ministral-3B** (or Llama 3.2) | GPU / Neural Engine (RAM: ~3GB) |
| **TTS** | Speaking (Text-to-Speech) | **Kokoro-82M** (ONNX) | CPU / GPU (Fast & High Quality) |
| **Orchestrator**| Pipeline Management | **Python** (AsyncIO) | CPU |

## Data Flow

1.  **Input Stream**:
    *   `fastrtc` captures 24kHz audio from the microphone.
    *   **VAD** analyzes audio buffer in real-time.
    *   When *Silence* is detected (> X ms), the buffer is "committed".

2.  **Perception (Hearing)**:
    *   Committed audio buffer $\rightarrow$ **Faster-Whisper**.
    *   Output: `User Transcript` (Text).

3.  **Cognition (Thinking)**:
    *   `User Transcript` + `System Prompt` $\rightarrow$ **Ollama API** (`/v1/chat/completions`).
    *   **Tool Calling**: If the LLM output matches a tool pattern (e.g., uses the `camera` tool), the app intercepts it, executes the Python function, and feeds the result back to the LLM.
    *   Output: `Assistant Response` (Text Stream).

4.  **Expression (Speaking & Moving)**:
    *   `Assistant Response` is buffered by sentence.
    *   Sentence $\rightarrow$ **Kokoro TTS** $\rightarrow$ Audio Waveform.
    *   **Tone Analysis**: The app scans the text for emotion tags (implied or explicit) to adjust robot gestures.
    *   **Playback**: Audio is streamed to speakers.
    *   **Synchronization**: Audio stream $\rightarrow$ `HeadWobbler` $\rightarrow$ Robot Head Motors (lipsync illusion).

## Directory Structure

We will introduce a `local/` module to keep this distinct from the cloud implementation:

```
src/reachy_mini_conversation_app/
├── local/
│   ├── __init__.py
│   ├── handler.py          # The main LocalSessionHandler (FastRTC compatible)
│   ├── vad.py              # Voice Activity Detection wrapper
│   ├── stt.py              # Faster-Whisper wrapper
│   ├── tts.py              # Kokoro wrapper
│   └── llm.py              # Ollama client wrapper
```

## Requirements

1.  **Ollama** installed and running (`ollama serve`).
2.  **Model pulled**: `ollama pull ministral-3b` (or your preferred model).
3.  **Python Packages**: `faster-whisper`, `kokoro-onnx`, `soundfile`, `scipy`.

```