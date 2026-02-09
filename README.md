---
title: Baby Reachy-Mini Companion
emoji: 🤖🍼
colorFrom: red
colorTo: blue
sdk: gradio
pinned: false
short_description: A fully local Reachy Mini AI Companion for babies and kids.
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Baby Reachy-Mini Companion

A fully local Reachy Mini AI Companion for babies and kids.

This "Dual-Interface" companion can interact with you both **locally via voice** (in the room) and **remotely via Signal** text messages. It features advanced audio perception, including baby cry detection, and uses your local LLM as a vision model.

<img src="docs/assets/baby-reachy-mini.jpg" width="600" alt="Baby Reachy Mini Companion" />

## Key Features

- **Omni-Channel Interaction:** Talk to Reachy naturally in the room, or text it via Signal when you're away.
- **Baby Monitor Mode:** Continuously listens for baby cries using an on-device audio classifier (YAMNet). If a cry is detected, it automatically soothes the baby and sends you a Signal alert.
- **Smart Sound Detection:** Beyond crying, Reachy can detect and react to other environmental sounds (like coughing, laughing, or shouting), enabling context-aware interactions.
- **Local Vision (VLM):** Uses your local multimodal LLM (like Qwen 2.5 VL via Ollama) to see and describe the world through the `camera` tool.
- **Neural Speech:** High-quality TTS via `Kokoro` (ONNX) and fast STT via `faster-whisper`.
- **Smart Motion:** Integrated head tracking (YOLO or MediaPipe), expressive dances, and emotional gestures.
- **Privacy First:** All processing—voice, vision, and chat—happens locally on your device.

## Installation

> [!IMPORTANT]
> Before using this app, ensure you have installed the [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini/).

### 1. Prerequisites

*   **Local LLM Server:** Install [Ollama](https://ollama.com/) (or any OpenAI-compatible server like vLLM) and pull a model:
    ```bash
    ollama pull qwen2.5:3b
    ```
*   **System Dependencies (macOS):**
    ```bash
    brew install portaudio gobject-introspection cairo pkg-config
    ```

### 2. Install

#### With uv (recommended)
```bash
# Install everything (audio, vision, wireless)
uv sync --extra local

# Or install only what you need
uv sync --extra local_audio             # Voice pipeline only
uv sync --extra local_audio --extra yolo_vision  # Voice + YOLO tracking
```

#### With pip
```bash
pip install -e ".[local]"          # Everything
pip install -e ".[local_audio]"    # Voice pipeline only
```

#### Optional dependency groups

| Extra | What it provides |
|-------|-----------------|
| `local_audio` | VAD, STT, TTS — the core voice pipeline (torch, silero-vad, faster-whisper, kokoro-onnx) |
| `yolo_vision` | YOLO-based face tracking (ultralytics, supervision) |
| `mediapipe_vision` | MediaPipe-based face tracking (lighter alternative to YOLO) |
| `reachy_mini_wireless` | GStreamer wireless support (PyGObject, gst-signalling) |
| `local` | All of the above combined |

### 3. Configure

Copy `.env.example` to `.env` and adjust the settings:

```bash
cp .env.example .env
```

The app connects to any **OpenAI-compatible** LLM server. By default it points to Ollama at `http://localhost:11434/v1`. If you use vLLM or another server, just change `LOCAL_LLM_URL` in your `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_LLM_URL` | `http://localhost:11434/v1` | URL to your OpenAI-compatible LLM server. |
| `LOCAL_LLM_MODEL` | `qwen2.5:3b` | Model name as known by your LLM server. |
| `LOCAL_LLM_API_KEY` | `ollama` | API key (Ollama ignores this; other servers may require a real key). |
| `LOCAL_STT_MODEL` | `small.en` | Whisper model size (`tiny.en`, `small.en`, `medium.en`, `large-v3`). |
| `MIC_GAIN` | `1.0` | Digital gain for microphone input (e.g., `2.0` to double volume). |
| `SIGNAL_USER_PHONE` | — | Your phone number (e.g., `+1234567890`) for remote alerts. |
| `REACHY_MINI_CUSTOM_PROFILE` | `default` | Selects the personality profile. |

## Running the App

```bash
# Simplest — uses settings from .env
uv run reachy-mini-conversation-app

# With YOLO face tracking
uv run reachy-mini-conversation-app --head-tracker yolo

# With MediaPipe face tracking (lighter)
uv run reachy-mini-conversation-app --head-tracker mediapipe

# Open the Gradio web UI
uv run reachy-mini-conversation-app --gradio

# Use OpenAI Realtime API instead of local processing
uv run reachy-mini-conversation-app --openai-realtime
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--head-tracker {yolo,mediapipe,None}` | Choose face-tracking backend (default: `None`). |
| `--no-camera` | Disable camera usage entirely. |
| `--smolvlm` | Use SmolVLM local vision model for periodic scene description. |
| `--gradio` | Open the Gradio web interface. |
| `--debug` | Enable debug logging. |
| `--robot-name NAME` | Zenoh topic prefix (only needed with multiple robots). |
| `--openai-realtime` | Use OpenAI Realtime API instead of local processing. |

### Signal Integration (Remote Access)
To enable the remote interface:
1.  **Install Signal-CLI:** `brew install signal-cli`
2.  **Register:** Link your account using `signal-cli link -n "Reachy"` (see [signal-cli docs](https://github.com/AsamK/signal-cli)).
3.  The app will automatically poll for messages. You can text "What do you see?" and Reachy will reply with a photo description!

## Troubleshooting

- **"Connection refused" from LLM** — Make sure your Ollama (or other LLM server) is running and the `LOCAL_LLM_URL` in `.env` is correct.
- **Slow first response** — The app pre-warms the LLM on startup. If using Ollama, the first model load can be slow; subsequent requests are fast.
- **STT too slow or inaccurate** — Try a different `LOCAL_STT_MODEL`. `tiny.en` is fastest, `medium.en` is most accurate, `small.en` is a good balance.
- **No audio input** — Check `MIC_GAIN` in `.env`. On some systems you may need to increase it (e.g., `2.0` or `3.0`).
- **MediaPipe/YOLO import errors** — Make sure you installed the right extra: `uv sync --extra mediapipe_vision` or `uv sync --extra yolo_vision`.

## Capabilities & Tools

The assistant is equipped with a suite of tools it can autonomously use:

| Tool | Action |
|------|--------|
| `camera` | Takes a picture and analyzes it using the local VLM (e.g., "What do you see?"). |
| `soothe_baby` | **Baby Monitor:** Performs gentle rocking motions and plays a soothing lullaby script. Triggered automatically by cry detection or manually. |
| `story_time` | Recites famous children's stories (Three Little Pigs, Goldilocks) with expressive narration. |
| `speak` | Explicitly speaks text (useful for precise multi-step tasks). |
| `dance` | Performs a dance move (e.g., `pendulum_swing`, `side_to_side_sway`). |
| `play_emotion` | Expresses emotions via antennas (happy, sad, surprised, etc.). |
| `move_head` | Moves the head to look in a specific direction. |
| `head_tracking` | Enables/disables face tracking. |
| `send_signal` | Sends a text message to your phone. |
| `send_signal_photo` | Takes a photo and sends it immediately to your phone via Signal. |

## Customization

### Profiles
You can change the assistant's personality in `src/reachy_mini_conversation_app/profiles/`.
*   `instructions.txt`: System prompt (personality, rules).
*   `tools.txt`: Enabled tools for this profile.

### Audio Event Detection
The app automatically downloads and runs a **YAMNet** audio classifier. It constantly listens for specific audio events to trigger autonomous actions:

*   **Baby Crying:** "Baby cry, infant cry", "Crying, sobbing", "Whimper" → *Triggers soothing mode.*
*   **Human Interactions:** "Laughter", "Coughing" → *Can trigger empathetic responses (e.g., "Are you okay?" or giggling back).*
*   **Alarms:** "Smoke detector", "Fire alarm" → *Can trigger urgent alerts.*

If detected, it triggers a system event that forces the LLM to call appropriate tools (like `soothe_baby`) and alert you via Signal.

## License
Apache 2.0
