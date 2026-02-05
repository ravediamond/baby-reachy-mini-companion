---
title: Reachy Mini Companion
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: A local, conversational companion for Reachy Mini.
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Mini Conversation App (Local Edition)

A fully local conversational assistant for the Reachy Mini robot, designed to be a "Dual-Interface" companion. It can interact with you both locally via **voice** and remotely via **Signal** text messages.

Powered by on-device LLMs, neural TTS, and speech recognition, it is designed for privacy, low latency, and multimodal capabilities.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Overview
- **Omni-Channel:** Talk to Reachy via voice in the room or via Signal message when away.
- **Local Intelligence:** Runs entirely on-device (Mac M-series or NVIDIA Jetson recommended).
- **Vision:** Uses your local LLM as a VLM (Vision Language Model) to "see" via the camera tool.
- **Speech:** `faster-whisper` for STT and `Kokoro` (ONNX) for high-quality neural voice synthesis.
- **Motion:** Integrated head tracking (YOLO or MediaPipe), dances, and emotional expressions.

## Installation

> [!IMPORTANT]
> Before using this app, you need to install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/).

### Prerequisites
1.  **Ollama:** Install [Ollama](https://ollama.com/) and pull a vision-capable model:
    ```bash
    ollama pull qwen2.5:3b # or your preferred VLM
    ```
2.  **System Dependencies (macOS):**
    ```bash
    brew install portaudio gobject-introspection cairo pkg-config
    ```

### Using uv (Recommended)
You can set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
# Install all local AI and vision dependencies
uv sync --extra local
```

## Configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_LLM_MODEL` | `qwen2.5:3b` | The model to use for both conversation and vision tasks. |
| `LOCAL_STT_MODEL` | `medium.en` | Whisper model size (`tiny.en`, `base.en`, `small.en`, `medium.en`). |
| `MIC_GAIN` | `1.0` | Boost microphone input (e.g., `2.0` to double volume). |
| `SIGNAL_USER_PHONE` | - | Your phone number for Signal remote interaction. |
| `REACHY_MINI_CUSTOM_PROFILE` | `default` | Selects the personality profile. |

## Running the App

```bash
# Run with local LLM (Ollama)
uv run reachy-mini-conversation-app --local-llm http://localhost:11434/v1

# Run with YOLO face tracking enabled
uv run reachy-mini-conversation-app --local-llm http://localhost:11434/v1 --head-tracker yolo
```

### Signal Integration (Remote Access)
1.  **Install Signal-CLI:** `brew install signal-cli`
2.  **Register:** Follow [signal-cli docs](https://github.com/AsamK/signal-cli) to link your account.
3.  The app will automatically poll for messages and respond using the LLM.

## LLM Tools

Reachy can use these tools to interact with the world:

| Tool | Action |
|------|--------|
| `camera` | Takes a picture and describes it (uses the local VLM). |
| `speak` | Explicitly speaks a piece of text (useful for multi-step tasks). |
| `story_time` | Recites famous children's stories with expressive narration. |
| `soothe_baby` | Gentle rocking motions combined with soothing lullaby words. |
| `dance` | Performs a dance move (e.g., `pendulum_swing`, `side_to_side_sway`). |
| `play_emotion` | Performs a scripted emotion (happy, sad, etc.). |
| `move_head` | Look in a specific direction. |
| `send_signal` | Sends a text message to your phone. |
| `send_signal_photo` | Takes a photo and sends it to your phone via Signal. |


## Custom Profiles

You can create custom personalities in `src/reachy_mini_conversation_app/profiles/`.
Each profile needs:
- `instructions.txt`: The system prompt for the LLM.
- `tools.txt`: A list of enabled tools.

To use a profile, set `REACHY_MINI_CUSTOM_PROFILE=your_profile_name`.

## License
Apache 2.0
