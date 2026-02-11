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

<video src="docs/assets/reachy-demo.mp4" width="600" controls>
  <a href="docs/assets/reachy-demo.mp4">Watch the demo video</a>
</video>

## Why I Built This

I'm a new dad on a mission: building a nursery companion that actually respects our privacy. I wanted a "cool nanny bot" that plays and helps out with the baby — without sending a single byte of data to the cloud. What happens at home stays at home.

Beyond this project, I want to prove that high-end robotics can run on consumer hardware (a Mac or a Jetson Orin) instead of massive servers or cloud subscriptions. If companion robotics only works on expensive cloud platforms, adoption will stay limited to tech demos. Running locally on a $200 Jetson or a Mac anyone already owns is how this technology actually reaches homes.

### On AI Companions and Children

People have asked me about the effects of AI companions and robots on children — and honestly, I fully agree that it's a delicate subject. But I believe we should explore it *because* it is something that will come one way or another. It's better to build an open, transparent experience where parents have full control than to wait for a closed commercial product that may not have their best interests at heart.

A few things that inform the design:

- **Privacy is non-negotiable.** Something running in your home, around your child, should never send data to a third party. That's why this is 100% local.
- **Physical safety is addressed by design.** Reachy Mini is a social robot with no hands and no manipulators — it can express, move its head, and communicate, but it cannot grab, push, or physically interact with the baby. Its antennas are used solely for emotional expression. The risk is minimal by the nature of the hardware.
- **Empathy is the key to acceptance.** A robot that executes tasks while a human is suffering has failed its purpose. One of my core goals is to explore giving the robot genuine empathetic behavior — detecting distress, adjusting tone, soothing rather than ignoring. When a companion robot truly acknowledges what you're feeling, that's when it becomes something people will accept in their lives.

## What Makes This Different

> **100% Local. No cloud. No exceptions.**
>
> The only Reachy Mini application running a fully local AI stack — LLM, speech-to-text, text-to-speech, vision, and audio classification — all on-device with zero cloud dependency.

- **7+ AI models on-device**: VAD, STT, LLM, TTS, VLM, YOLO, and YAMNet — orchestrated together in a single pipeline
- **Autonomous safety**: Detects baby cries and soothes automatically. YOLO continuously scans for dangerous objects near the baby and triggers a VLM analysis with a Signal photo alert to the parent — all decided by the LLM, not a script
- **Full Reachy Mini integration**: Camera, head motion (100Hz control loop), antenna emotions, dances, face tracking, and Reachy Mini Apps headless mode
- **NVIDIA Jetson native**: Runs on Jetson Orin using NVIDIA's official AI containers (`ghcr.io/nvidia-ai-iot/vllm`) with GPU-accelerated vLLM inference, quantized models tuned for Jetson's memory bandwidth, and YOLO on GPU for real-time vision

## A Complete Companion

An all-in-one solution for your child — no screens required:

- **Entertain** — Play, chat, and interact through voice conversations, dances, and expressive movements. Screen-free engagement that keeps your child naturally entertained.
- **Soothe & Sleep** — Sing lullabies and nursery rhymes to calm or put your baby to sleep. Gentle rocking motions and soft speech for a soothing bedtime routine.
- **Spark Imagination** — Tell stories tailored to your child's world. Pick characters, animals, or themes and let the companion weave them into an adventure.
- **Learn & Discover** — Recite the alphabet, explore animals, discover the solar system — interactive lessons adapted to your child's curiosity and pace.

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
# Base install — includes the full voice pipeline (STT, TTS, VAD)
uv sync

# Add vision extras as needed
uv sync --extra yolo_vision        # + YOLO face tracking & danger detection
uv sync --extra local              # Everything (YOLO, MediaPipe, wireless)
```

#### With pip
```bash
pip install -e "."                 # Voice pipeline
pip install -e ".[local]"          # Everything
```

#### What's included in the base install

The core voice pipeline — **faster-whisper** (STT), **Kokoro** (TTS), **Silero VAD**, and **PyTorch** — is included in the base dependencies. This means the app works out of the box from the Reachy Mini app store without needing optional extras.

#### Optional dependency groups

| Extra | What it provides |
|-------|-----------------|
| `yolo_vision` | YOLO-based face tracking and danger detection (ultralytics, supervision) |
| `mediapipe_vision` | MediaPipe-based face tracking (lighter alternative to YOLO) |
| `reachy_mini_wireless` | GStreamer wireless support (PyGObject, gst-signalling) |
| `local` | All of the above combined |

#### A note on PyTorch

PyTorch (~2 GB) is a base dependency because **Silero VAD** requires it — the `silero-vad` pip package lists `torch` as a hard dependency, and even its ONNX code path uses `torch.cat`, `torch.zeros`, and `torch.from_numpy` internally for state management around the ONNX inference call.

In theory, you could eliminate PyTorch entirely by writing a custom numpy-only ONNX wrapper (the torch operations are trivial array ops) and loading the Silero ONNX model directly with `onnxruntime`. We chose not to because:

- It means not using the `silero-vad` package at all — you'd download the ONNX model yourself and maintain a fork of their inference code
- PyTorch installs fine on the target hardware (Mac M1, NVIDIA Jetson)
- The other heavy deps (`faster-whisper`, `kokoro-onnx`) already pull in `onnxruntime`, so the incremental cost is just torch itself

### 3. Configure

There are two ways to configure the app depending on how you run it:

#### Option A: Via `.env` file (standalone CLI usage)

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

#### Tested Models

The following models have been tested and work well with the app:

| Model | Type | Notes |
|-------|------|-------|
| `qwen2.5:3b` | Text LLM | **Recommended default.** Stable on Ollama and vLLM. Great tool-calling support for its size. |
| `ministral:3b` (Mistral AI) | Text LLM | Excellent reasoning for a 3B model — frontier-level quality in a tiny footprint. Works on Ollama. Does **not** work on vLLM v0.14 (current Jetson container version). |
| `qwen2.5-vl:3b` | Vision LLM | Enables the camera tool for visual questions. Stable on Ollama and vLLM. |
| `qwen3-vl:4b` | Vision LLM | Better quality than qwen2.5-vl, but the Qwen3 series uses a dynamic patching system that requires the **latest** inference engine versions — may not work on older Ollama/vLLM builds. |

> [!NOTE]
> Newer models (Ministral 3B, Qwen3-VL) deliver better results but can have compatibility issues with older inference engines. If you're on Jetson with vLLM v0.14, stick with `qwen2.5:3b` or `qwen2.5-vl:3b`. On a Mac with the latest Ollama, all models work.

#### Option B: Via the Settings UI (Reachy Mini Apps)

When the app is launched through the **Reachy Mini Apps** system (headless mode), it exposes a web-based settings page instead of reading a `.env` file directly.

1. The app starts and opens a **configuration page** in your browser.
2. Fill in the LLM settings (Server URL, Model name, API key, STT model).
3. Click **"Start"** — the pipeline initializes with your chosen settings.
4. Settings are persisted to the app's instance directory, so they are remembered across restarts.

On subsequent launches, the saved settings are pre-populated in the form. You can review and adjust them before clicking Start again.

> [!NOTE]
> In OpenAI Realtime mode (`--openai-realtime`), the settings page shows an API key field instead. In local mode (default), it shows the full LLM configuration form.

## Running the App

### Starting the Daemon

Before running the conversation app, start the Reachy Mini daemon with `--deactivate-audio`. This prevents the daemon from monopolizing the microphone — the conversation app needs direct access to capture audio for speech recognition.

```bash
# Mac (simulation mode)
uv run reachy-mini-daemon --sim --deactivate-audio

# Jetson (physical robot)
uv run reachy-mini-daemon --serialport /dev/ttyACM0 --deactivate-audio
```

> [!IMPORTANT]
> Without `--deactivate-audio`, the daemon's internal audio handling captures all microphone input, and the conversation app receives only silence. This flag is **required** for the conversation app to work.

### Standalone (CLI)

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

### As a Reachy Mini App (headless)

When installed as a Reachy Mini App (via the `reachy_mini_apps` entry point), the app is discovered and launched automatically by the Reachy Mini daemon. In this mode:

1. The app serves a settings UI at its `custom_app_url` (`http://0.0.0.0:7860/`).
2. The pipeline **waits** for you to configure and click **Start** before initializing.
3. Once running, the settings page shows the active model and provides access to the personality studio.

No `.env` file is needed — all configuration happens through the browser.

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

## Architecture

<img src="docs/assets/conversation_app_arch.svg" width="800" alt="Architecture — fully local AI pipeline" />

The entire pipeline runs on-device: audio is captured and processed through VAD, STT, and the LLM with tool calling. The LLM autonomously invokes tools (camera, motion, Signal alerts) based on context. TTS output is streamed back with audio-reactive head movement for natural-looking speech.

## Deployment Scenarios

The app supports three deployment configurations depending on your hardware.

### 1. Everything on Mac (default)

The simplest setup. Ollama runs the LLM, and the app handles STT/TTS locally on the Mac.

```bash
ollama pull qwen2.5:3b
uv run reachy-mini-conversation-app
```

Set in `.env`:
```env
LOCAL_LLM_URL="http://localhost:11434/v1"
LOCAL_LLM_MODEL="qwen2.5:3b"
```

### 2. App on Mac, vLLM on Jetson Orin (hybrid)

Run the app and audio pipeline on the Mac, but offload LLM inference to a Jetson Orin running [vLLM](https://docs.vllm.ai/) for GPU-accelerated inference.

**On the Jetson**, enable max performance and start vLLM:
```bash
# Enable max performance on Jetson
sudo nvpmodel -m 0
sudo jetson_clocks

# Start vLLM via Docker (text-only, quantized for speed)
docker run --rm -it --runtime nvidia --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve RedHatAI/Qwen3-4B-quantized.w4a16 \
  --served-model-name qwen --port 30000 --dtype half \
  --trust-remote-code --gpu-memory-utilization 0.80 \
  --max-model-len 4096 --max-num-seqs 2
```

**On the Mac**, create an SSH tunnel to forward the vLLM port:
```bash
ssh -L 30000:localhost:30000 user@<jetson-ip>
```

Then set in `.env`:
```env
LOCAL_LLM_URL="http://localhost:30000/v1"
LOCAL_LLM_MODEL="qwen"
LOCAL_LLM_API_KEY="token-abc123"
```

Run the app normally on Mac:
```bash
uv run reachy-mini-conversation-app
```

### 3. Everything on Jetson Orin

Run the full app directly on the Jetson Orin, including STT, TTS, LLM, and robot control.

#### Jetson performance setup
```bash
# Enable max performance mode (REQUIRED for good inference speed)
sudo nvpmodel -m 0
sudo jetson_clocks
```

#### Why quantized models matter on Jetson

The Jetson Orin NX has **limited memory bandwidth** (~102 GB/s) compared to desktop GPUs. Since LLM inference is memory-bandwidth-bound (loading model weights from VRAM for each token), **quantized models are essential** for acceptable speed:

- **FP16** (full precision): Large memory footprint, limited by bandwidth. Slower on Jetson.
- **W4A16** (4-bit weights, 16-bit activations): ~4x less memory, ~4x faster token throughput. The sweet spot for Jetson.
- **AWQ-4bit**: Similar to W4A16, hardware-aware quantization with minimal quality loss.

A 4B model at W4A16 fits comfortably in the Orin NX's 16GB and runs at ~30 tok/s. The same model at FP16 would be significantly slower and may not fit at all.

#### LLM server — vLLM via Docker (recommended)

The NVIDIA Jetson container for vLLM provides a pre-built image optimized for the Orin architecture. The official container ships with **vLLM v0.11**. A newer build with **vLLM v0.14** is also available and provides better performance and model support — check [NVIDIA's Jetson AI containers](https://github.com/dusty-nv/jetson-containers) for the latest.

**Text-only (fastest):**
```bash
docker run --rm -it --runtime nvidia --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve RedHatAI/Qwen3-4B-quantized.w4a16 \
  --served-model-name qwen --port 30000 --dtype half \
  --trust-remote-code --gpu-memory-utilization 0.80 \
  --max-model-len 4096 --max-num-seqs 2
```

**Vision + Text (Qwen3-VL):**
```bash
docker run --rm -it --runtime nvidia --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve cpatonn/Qwen3-VL-4B-Instruct-AWQ-4bit \
  --served-model-name qwen --port 30000 --dtype half \
  --trust-remote-code --gpu-memory-utilization 0.8 \
  --max-model-len 1024 --limit-mm-per-prompt '{"image": 1}' \
  --max-num-seqs 1
```

Set in `.env`:
```env
LOCAL_LLM_URL="http://localhost:30000/v1"
LOCAL_LLM_MODEL="qwen"
LOCAL_LLM_API_KEY="token-abc123"
MIC_GAIN=5000.0
```

> [!IMPORTANT]
> **Microphone on Jetson**: The ALSA driver reports very low mic levels. You **must** set `MIC_GAIN` to a high value (e.g., `5000.0`). Without this, the VAD will never trigger.

> [!IMPORTANT]
> **Daemon audio conflict**: The Reachy Mini daemon's internal audio handling can interfere with the microphone on Jetson. Start the daemon with audio deactivated:
> ```bash
> uv run reachy-mini-daemon --serialport /dev/ttyACM0 --deactivate-audio
> ```

#### LLM engine benchmarks on Jetson Orin NX (16GB)

| Engine | Model | Quantization | TPS | Notes |
|--------|-------|-------------|-----|-------|
| **vLLM v0.11** | Qwen3-4B | W4A16 | ~29 | Docker container, text-only |
| **vLLM v0.11** | Qwen3-VL-4B | AWQ-4bit | ~28 (text) / ~8 (vision) | Same container, multimodal |
| **vLLM v0.14** | Qwen3-4B | W4A16 | Better | Latest Jetson build, improved scheduling |
| **llama.cpp** | Qwen2.5-3B | GGUF Q4_K_M | ~23 | Built natively for Jetson aarch64. Functional but slower than vLLM for equivalent models due to less GPU optimization. |
| **Ollama** | qwen2.5:3b | Q4_K_M | Similar to llama.cpp | Easy install, wraps llama.cpp internally |

> [!TIP]
> **Use quantized models** (W4A16, AWQ-4bit, GGUF Q4) on Jetson. The memory bandwidth bottleneck means FP16 models run much slower with no meaningful quality gain for conversational use cases.

#### RAM management

If the Jetson runs out of memory after stopping vLLM:
```bash
docker stop $(docker ps -q)
docker system prune -f && sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

## Troubleshooting

### General
- **Microphone returns silence / no speech detected** — The most common issue. Make sure the daemon is started with `--deactivate-audio`. Without this flag, the daemon captures all microphone input internally and the app receives only zeros. See [Starting the Daemon](#starting-the-daemon).
- **"Connection refused" from LLM** — Make sure your Ollama (or other LLM server) is running and the URL is correct (check `.env` or the settings UI).
- **Slow first response** — The app pre-warms the LLM on startup. If using Ollama, the first model load can be slow; subsequent requests are fast.
- **STT too slow or inaccurate** — Try a different STT model. `tiny.en` is fastest, `medium.en` is most accurate, `small.en` is a good balance.
- **No audio input after confirming `--deactivate-audio`** — Check `MIC_GAIN` in `.env`. On some systems you may need to increase it (e.g., `2.0` or `3.0`).
- **MediaPipe/YOLO import errors** — Make sure you installed the right extra: `uv sync --extra mediapipe_vision` or `uv sync --extra yolo_vision`.
- **Settings not taking effect** — In headless mode, settings are applied when you click Start. If already running, they take effect on the next restart.

### Jetson Orin
- **Microphone not working** — The most common issue. Set `MIC_GAIN=5000.0` in `.env`. The Jetson ALSA driver reports near-silent mic levels by default, so the VAD never detects speech.
- **Speaker volume too low** — ALSA output is also quiet on Jetson. You may need to adjust ALSA mixer levels with `alsamixer` or `amixer`.
- **Mic captured by daemon** — Start the Reachy Mini daemon with `--deactivate-audio` so the app can control audio directly.
- **Wrong audio device selected** — Jetson often has multiple sound cards. Check which one is "Reachy Mini Audio" with `arecord -l` and ensure PulseAudio is not grabbing it.
- **PulseAudio interference** — If PulseAudio is running, it can monopolize the audio device. Disable it: `systemctl --user stop pulseaudio.socket pulseaudio.service`.

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
| `check_danger` | **Visual Safety Monitor:** Queries the YOLO-based danger detector for hazardous objects (scissors, knives, forks) near the baby. |
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

### Visual Safety Detection
When YOLO vision is installed (`uv sync --extra yolo_vision`), the app runs a **continuous danger detector** alongside the camera feed. Every 2 seconds it scans for hazardous objects using a general-purpose YOLO model:

*   **Dangerous objects:** Scissors, knives, forks → *YOLO detects the object, then triggers a VLM camera analysis for confirmation, and sends a Signal photo alert to the parent.*
*   **Two-stage pipeline:** Fast YOLO detection (low compute) acts as a trigger for expensive VLM analysis (high accuracy), keeping GPU usage efficient.
*   **Throttled:** Alerts are rate-limited to once per 30 seconds to prevent notification spam.

If detected, it triggers a system event that forces the LLM to call appropriate tools (like `soothe_baby`) and alert you via Signal.

## License
Apache 2.0
