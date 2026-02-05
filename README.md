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

A fully local, multimodal conversational assistant for the Reachy Mini robot. 

This "Dual-Interface" companion can interact with you both **locally via voice** (in the room) and **remotely via Signal** text messages. It features advanced audio perception, including baby cry detection, and uses your local LLM as a vision model.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Key Features

- **Omni-Channel Interaction:** Talk to Reachy naturally in the room, or text it via Signal when you're away.
- **Baby Monitor Mode:** Continuously listens for baby cries using an on-device audio classifier (YAMNet). If a cry is detected, it automatically soothes the baby and sends you a Signal alert.
- **Local Vision (VLM):** Uses your local multimodal LLM (like Qwen 2.5 VL via Ollama) to see and describe the world through the `camera` tool.
- **Neural Speech:** High-quality TTS via `Kokoro` (ONNX) and fast STT via `faster-whisper`.
- **Smart Motion:** Integrated head tracking (YOLO or MediaPipe), expressive dances, and emotional gestures.
- **Privacy First:** All processing—voice, vision, and chat—happens locally on your device.

## Installation

> [!IMPORTANT]
> Before using this app, ensure you have installed the [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini/).

### 1. Prerequisites
*   **Ollama:** Install [Ollama](https://ollama.com/) and pull a vision-capable model (recommended for full VLM features):
    ```bash
    ollama pull qwen2.5-vl:3b  # Or your preferred VLM
    ```
*   **System Dependencies (macOS):**
    ```bash
    brew install portaudio gobject-introspection cairo pkg-config
    ```

### 2. Project Setup
We recommend using `uv` for fast dependency management:

```bash
# Install all dependencies (audio, vision, wireless support)
uv sync --extra local
```

## Configuration

Copy `.env.example` to `.env` and adjust the settings:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_LLM_MODEL` | `qwen2.5:3b` | The model used for both chat and vision (VLM). |
| `LOCAL_LLM_URL` | `http://...` | URL to your local LLM server (default: Ollama). |
| `LOCAL_STT_MODEL` | `medium.en` | Whisper model size (`tiny`, `small`, `medium`, `large`). |
| `MIC_GAIN` | `1.0` | Digital gain for microphone input (e.g., `2.0` to double volume). |
| `SIGNAL_USER_PHONE` | - | Your phone number (e.g., `+1234567890`) for remote alerts. |
| `REACHY_MINI_CUSTOM_PROFILE` | `default` | Selects the personality profile. |

## Running the App

### Standard Mode (Voice + Vision + Tracking)
```bash
# Run with YOLO face tracking (robust)
uv run reachy-mini-conversation-app --local-llm http://localhost:11434/v1 --head-tracker yolo

# Run with MediaPipe face tracking (lightweight/fast)
uv run reachy-mini-conversation-app --local-llm http://localhost:11434/v1 --head-tracker mediapipe
```

### Signal Integration (Remote Access)
To enable the remote interface:
1.  **Install Signal-CLI:** `brew install signal-cli`
2.  **Register:** Link your account using `signal-cli link -n "Reachy"` (see [signal-cli docs](https://github.com/AsamK/signal-cli)).
3.  The app will automatically poll for messages. You can text "What do you see?" and Reachy will reply with a photo description!

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
The app automatically downloads and runs a **YAMNet** audio classifier. It constantly listens for:
*   "Baby cry, infant cry"
*   "Crying, sobbing"
*   "Whimper"

If detected, it triggers a system event that forces the LLM to call `soothe_baby` and alert you via Signal.

## License
Apache 2.0