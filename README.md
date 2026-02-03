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

A fully local conversational app for the Reachy Mini robot, powered by on-device LLMs, neural TTS, and speech recognition. Designed for privacy, low latency, and offline capability.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Architecture

The app follows a layered architecture connecting the user, local AI services, and robot hardware:

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## Overview
- **Local Intelligence:** Runs entirely on your machine (Mac M-series or NVIDIA Jetson recommended).
- **Speech Recognition:** Uses `faster-whisper` for fast, accurate speech-to-text.
- **Text-to-Speech:** Uses `Kokoro` (ONNX) for high-quality, lightweight neural voice synthesis.
- **Brain:** Compatible with local LLMs via [Ollama](https://ollama.com/) (e.g., Qwen 2.5, Ministral).
- **Vision:** Supports local vision models (SmolVLM2) or Ollama vision models.
- **Motion:** Layered motion system for dances, emotions, and face tracking.

## Installation

> [!IMPORTANT]
> Before using this app, you need to install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/).

### Prerequisites
1.  **Ollama:** Install [Ollama](https://ollama.com/) and pull your desired model:
    ```bash
    ollama pull qwen2.5:3b
    ```
2.  **System Dependencies:** Ensure you have `portaudio` installed (e.g., `brew install portaudio` on Mac).

### Using uv (Recommended)
You can set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
# Install dependencies including local Mac/Linux optimizations
uv sync --extra local_mac
```

### macOS Camera Fix (Dark Image)
If the camera image appears very dark on macOS, you may need to disable auto-exposure priority. You can use the provided script:

```bash
./scripts/fix_mac_camera.sh
```

Alternatively, you can do it manually:

1. **Install `uvc-util`**:
   ```bash
   git clone https://github.com/jtfrey/uvc-util.git
   cd uvc-util
   make
   ```

2. **Fix Exposure**:
   Run the following command:
   ```bash
   ./uvc-util -I 0x01140000 -s auto-exposure-priority=1
   ```
   *Note: If the command fails, use `./uvc-util -l` to list your devices and find the correct interface ID.*

## Configuration

Copy `.env.example` to `.env` or set environment variables directly.

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_LLM_MODEL` | `qwen2.5:3b` | The Ollama model to use for conversation. |
| `LOCAL_STT_MODEL` | `small.en` | Whisper model size (`tiny.en`, `base.en`, `small.en`, `medium.en`). |
| `LOCAL_VISION_MODEL` | `HuggingFaceTB/SmolVLM2...` | Path or ID for the vision model (or `ollama:llama3.2-vision`). |
| `REACHY_MINI_CUSTOM_PROFILE` | `default` | Selects the personality profile (e.g., `baby_buddy`). |

## Omni-Channel Agent (Audio + Signal)

This mode transforms Reachy into a "Dual-Interface" assistant that can interact with you both locally (via voice) and remotely (via Signal text messages). It uses a unified event loop and `smolagents` to intelligently decide whether to speak out loud or send a text message based on the input source and context.

**Features:**
- **Room Awareness:** Speaks to you when you talk to it in the room.
- **Remote Access:** Responds to Signal messages when you are away.
- **Smart Routing:** Can text you a shopping list even if you asked for it by voice, or speak to the room if you text it to "tell the baby something".

### Setup
1.  **Install Signal-CLI:**
    ```bash
    brew install signal-cli  # macOS
    # Register your account (see signal-cli docs)
    signal-cli -u +1234567890 register
    signal-cli -u +1234567890 verify <code-from-sms>
    ```
2.  **Install Python Dependencies:**
    ```bash
    uv pip install smolagents
    ```

### Usage
Run with the `--omni-agent` flag:

```bash
uv run reachy-mini-conversation-app --omni-agent --local-llm http://localhost:11434/v1
```

*Note: This mode requires a local LLM server (Ollama) to be running.*

## Running the App

Activate your virtual environment and launch the app. Ensure your Reachy Mini (or simulator) is on.

```bash
# Run with default settings
uv run reachy-mini-conversation-app --local-llm http://localhost:11434/v1

# Run with a specific profile and larger speech model
REACHY_MINI_CUSTOM_PROFILE=baby_buddy LOCAL_STT_MODEL="medium.en" uv run reachy-mini-conversation-app --local-llm http://localhost:11434/v1
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--local-llm <URL>` | URL to your local LLM server (e.g., `http://localhost:11434/v1`). Required for local mode. |
| `--omni-agent` | Enable the Omni-Channel Agent (Audio + Signal). Requires `smolagents`. |
| `--head-tracker {yolo,mediapipe}` | Enable face tracking (requires camera). |
| `--no-camera` | Disable camera usage (audio only). |
| `--debug` | Enable verbose logging. |

## LLM Tools

The assistant can control the robot using these tools:

| Tool | Action |
|------|--------|
| `move_head` | Look in a specific direction (up, down, left, right). |
| `play_emotion` | Perform a pre-scripted emotion (happy, sad, surprised, etc.). |
| `dance` | Perform a dance move. |
| `check_baby_status` | (Profile-specific) Analyze camera input for safety monitoring. |

## Custom Profiles

You can create custom personalities in `src/reachy_mini_conversation_app/profiles/`.
Each profile needs:
- `instructions.txt`: The system prompt for the LLM.
- `tools.txt`: A list of enabled tools.

To use a profile, set `REACHY_MINI_CUSTOM_PROFILE=your_profile_name`.

## License
Apache 2.0
