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

Beyond this project, I want to prove that high-end robotics can run on consumer hardware — a Mac for the app and a $200 Jetson Orin for GPU inference — instead of massive servers or cloud subscriptions. If companion robotics only works on expensive cloud platforms, adoption will stay limited to tech demos. Running locally on hardware anyone can afford is how this technology actually reaches homes.

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
- **NVIDIA Jetson vLLM**: Offload LLM inference to a Jetson Orin running GPU-accelerated vLLM via NVIDIA's official AI containers, with quantized models tuned for Jetson's memory bandwidth

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

### Audio Architecture

Understanding how audio works helps choose the right setup for your platform.

The Reachy Mini SDK has its own audio layer: the daemon creates a `SoundDeviceAudio` backend that can open the robot's USB mic ("Reachy Mini Audio") for sound effects. Separately, the conversation app creates its **own** `SoundDeviceAudio` instance (via the `ReachyMini` SDK object passed by the runtime) and calls `media.start_recording()` to capture audio for speech recognition.

These are **independent audio pipelines** — the daemon uses audio for output only (wake-up/sleep sounds), while the app uses it for input (microphone recording). There is no audio sharing via Zenoh or shared memory between them.

**This works correctly on Linux** (the target platform), where ALSA's `dmix`/`dsnoop` plugins allow multiple processes to share the same USB audio device for simultaneous input and output.

**On macOS**, the Reachy Mini's USB audio interface does not reliably support concurrent access from multiple processes. When the daemon opens the device for output, the app's input stream on the same device may return silence (all zeros). This is a platform-specific limitation of the USB audio chip — the microphone hardware is fine, but macOS CoreAudio cannot share it across processes the way ALSA can.

#### Two audio modes

The settings dashboard provides a **microphone selector** that offers two approaches:

| Mode | How it works | When to use |
|------|-------------|-------------|
| **SDK audio** (default) | Uses `media.get_audio_sample()` from the Reachy Mini SDK | Linux (robot, Jetson) — works out of the box |
| **Direct mic** (via selector) | Bypasses the SDK and opens the chosen mic directly via SoundDevice | macOS development — select "MacBook Air Microphone" or any working input device |

When you select a specific microphone in the dashboard, the app opens it directly with `sounddevice.InputStream`, bypassing the SDK's audio layer entirely. This avoids the USB device sharing issue on macOS.

### Starting the Daemon

#### As a Reachy Mini App (recommended)

When installed as a Reachy Mini App (via the `reachy_mini_apps` entry point), the app is discovered and launched automatically by the Reachy Mini daemon. The daemon manages the lifecycle — no manual startup needed.

In this mode:

1. The app serves a settings UI at its `custom_app_url` (`http://0.0.0.0:7860/`).
2. The pipeline **waits** for you to configure LLM settings, select a microphone, and click **Start**.
3. Once running, the settings page shows the active model and provides access to the personality studio.

No `.env` file is needed — all configuration happens through the browser.

> [!NOTE]
> On macOS, use the **microphone selector** in the settings dashboard to pick your Mac's built-in mic (or another working input device). The default SDK audio path reads from "Reachy Mini Audio" which returns silence on macOS due to the USB device sharing limitation described above.

#### Standalone (manual daemon)

If you prefer to manage the daemon yourself, start it with `--deactivate-audio` so it doesn't open the USB audio device at all. This frees the device for the app to use directly.

```bash
# Mac (simulation mode)
uv run reachy-mini-daemon --sim --deactivate-audio

# Mac (physical robot connected via USB)
uv run reachy-mini-daemon --deactivate-audio

# Jetson (physical robot)
uv run reachy-mini-daemon --serialport /dev/ttyACM0 --deactivate-audio
```

> [!NOTE]
> `--deactivate-audio` disables the daemon's sound effects (wake-up/sleep sounds). It does **not** affect the conversation app's audio — the app handles its own recording and playback independently.

### Running the Conversation App

```bash
# Simplest — uses settings from .env
uv run reachy-mini-conversation-app

# With settings dashboard (opens browser)
uv run reachy-mini-conversation-app --dashboard

# With YOLO face tracking
uv run reachy-mini-conversation-app --head-tracker yolo

# With MediaPipe face tracking (lighter)
uv run reachy-mini-conversation-app --head-tracker mediapipe

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

## Architecture

<img src="docs/assets/conversation_app_arch.svg" width="800" alt="Architecture — fully local AI pipeline" />

The entire pipeline runs on-device: audio is captured and processed through VAD, STT, and the LLM with tool calling. The LLM autonomously invokes tools (camera, motion, Signal alerts) based on context. TTS output is streamed back with audio-reactive head movement for natural-looking speech.

## Deployment Scenarios

The app runs on a Mac (or any desktop). The Jetson Orin is used **only as a vLLM inference server** — the conversation app itself does not run on the Jetson.

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

### 2. App on Mac, vLLM on Jetson Orin

Run the app and audio pipeline on the Mac, but offload LLM inference to a Jetson Orin running [vLLM](https://docs.vllm.ai/) for GPU-accelerated inference. This gives you faster token generation than CPU-based Ollama while keeping the audio/robot pipeline on the Mac.

See [Setting Up vLLM on Jetson Orin](#setting-up-vllm-on-jetson-orin) for the full Jetson setup guide.

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

## Setting Up vLLM on Jetson Orin

This section covers running vLLM on a Jetson Orin NX (16GB) as a remote LLM inference server for the conversation app.

### Jetson preparation

#### Power mode

Set the Jetson to maximum performance mode. This is **required** for acceptable inference speed — the default power mode throttles the GPU significantly.

```bash
# Enable MAXN power mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

#### Headless operation

The Jetson typically runs headless (no display). Connect via SSH and manage everything from the terminal. If you need to adjust audio levels, note that `alsamixer` requires an interactive terminal — it will block in a non-interactive shell. Use `amixer` for scriptable audio control instead:

```bash
# Check audio devices
arecord -l

# Adjust mic level (scriptable, unlike alsamixer)
amixer -c 1 set Mic 80%
```

### Docker containers

NVIDIA provides pre-built Docker images for vLLM optimized for Jetson's architecture. Use these instead of building from source — they include the correct CUDA toolkit and are tested against Jetson's unified memory architecture.

The official container ships with **vLLM v0.11**. A newer community build with **vLLM v0.14.0** is available from [NVIDIA's Jetson AI containers](https://github.com/dusty-nv/jetson-containers) and provides better performance, improved scheduling, and broader model support. We recommend v0.14.0 when available.

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

### Why quantized models matter on Jetson

The Jetson Orin NX has **limited memory bandwidth** (~102 GB/s) compared to desktop GPUs. Since LLM inference is memory-bandwidth-bound (loading model weights from VRAM for each token), **quantized models are essential** for acceptable speed:

- **FP16** (full precision): Large memory footprint, limited by bandwidth. Slower on Jetson.
- **W4A16** (4-bit weights, 16-bit activations): ~4x less memory, ~4x faster token throughput. The sweet spot for Jetson.
- **AWQ-4bit**: Similar to W4A16, hardware-aware quantization with minimal quality loss.

A 4B model at W4A16 fits comfortably in the Orin NX's 16GB and runs at ~30 tok/s. The same model at FP16 would be significantly slower and may not fit at all.

> [!TIP]
> **Use quantized models** (W4A16, AWQ-4bit, GGUF Q4) on Jetson. The memory bandwidth bottleneck means FP16 models run much slower with no meaningful quality gain for conversational use cases.

### Monitoring and debugging

Use NVIDIA's tools to monitor GPU utilization, memory bandwidth, and thermal throttling:

```bash
# jtop — Jetson-specific system monitor (GPU, CPU, RAM, temps, power)
# Install: sudo pip3 install jetson-stats
jtop

# tegrastats — raw GPU/memory utilization from the terminal
sudo tegrastats --interval 1000

# Check vLLM is serving correctly
curl http://localhost:30000/v1/models
```

`jtop` is particularly useful for verifying that MAXN power mode is active and that the GPU is actually being utilized during inference. `tegrastats` shows real-time memory bandwidth usage — critical for understanding whether your quantized model is hitting the bandwidth ceiling.

### Benchmarks on Jetson Orin NX (16GB)

| Engine | Model | Quantization | TPS | Notes |
|--------|-------|-------------|-----|-------|
| **vLLM v0.11** | Qwen3-4B | W4A16 | ~29 | Docker container, text-only |
| **vLLM v0.11** | Qwen3-VL-4B | AWQ-4bit | ~28 (text) / ~8 (vision) | Same container, multimodal |
| **vLLM v0.14** | Qwen3-4B | W4A16 | Better | Latest Jetson build, improved scheduling |
| **llama.cpp** | Qwen2.5-3B | GGUF Q4_K_M | ~23 | Built natively for Jetson aarch64. Functional but slower than vLLM for equivalent models due to less GPU optimization. |
| **Ollama** | qwen2.5:3b | Q4_K_M | Similar to llama.cpp | Easy install, wraps llama.cpp internally |

### RAM management

The Jetson has 16GB of unified memory shared between CPU and GPU. If the Jetson runs out of memory after stopping vLLM:
```bash
docker stop $(docker ps -q)
docker system prune -f && sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

## Why Not Run Everything on Jetson?

We initially attempted to run the full conversation app (STT, TTS, VAD, YOLO, MediaPipe, audio pipeline, robot control) alongside vLLM on the Jetson Orin. It works — there is a [working Jetson-native version](cloned_repo_jetson/) in this repo — but the experience taught us it's not the right approach for this project. Here's why:

### Memory pressure

The Jetson Orin NX has **16GB of unified memory** shared between CPU and GPU. vLLM alone needs 8-10GB for a 4B quantized model. Adding the Python runtime, PyTorch, faster-whisper, YOLO, MediaPipe, and the Reachy Mini daemon leaves almost no headroom. The system swaps constantly, killing inference speed and making the whole experience sluggish.

### Python overhead

Each Python process (STT, TTS, YOLO, the app itself) loads its own interpreter, its own copies of NumPy/PyTorch, and its own model weights. On a desktop with 32-64GB this is fine. On a 16GB Jetson where every megabyte counts, it's wasteful. The conversation app alone can use 2-3GB of RAM before the LLM even loads.

### Audio on headless Jetson

ALSA on Jetson reports very low microphone levels by default, requiring extreme gain values (`MIC_GAIN=5000.0`). Adjusting ALSA mixer levels with `alsamixer` requires an interactive terminal — problematic on a headless device. PulseAudio can interfere with device access and needs to be disabled. These are solvable but add friction to an already constrained environment.

### The Reachy Mini daemon

The Reachy Mini daemon needs to run alongside the app for robot control (Zenoh communication). On Linux, ALSA's `dsnoop` allows audio sharing between the daemon and app, but the daemon itself consumes resources. Starting it correctly with the right serial port (`--serialport /dev/ttyACM0`) and managing its lifecycle on a headless Jetson adds operational complexity.

### Our conclusion

The Jetson Orin is excellent at **one thing**: GPU-accelerated LLM inference via vLLM in a Docker container. It does this better and cheaper than any cloud API. But trying to run 7+ AI models, a robot control daemon, audio processing, and an LLM server all on 16GB of unified memory creates a resource-constrained environment where everything runs worse.

The hybrid approach (app on Mac, vLLM on Jetson) plays to each device's strengths: the Mac handles audio, vision, and orchestration with plenty of RAM, while the Jetson focuses exclusively on fast LLM inference.

## Future: Optimized On-Device Deployment

Running everything on a Jetson *could* work well with the right architecture — but it requires moving away from Python-per-model toward a systems-level approach:

- **[dora-rs](https://dora-rs.ai/)** — A Rust-based robotics dataflow framework with zero-copy shared memory. Instead of each Python process holding its own copy of data, dora-rs nodes share tensors through memory-mapped buffers. On a unified memory device like the Jetson, this eliminates redundant copies entirely.
- **Rust daemon** — The Reachy Mini SDK includes a Rust-based daemon. Using it natively (instead of the Python wrapper) would eliminate one Python process and its memory overhead.
- **DLA cores** — The Jetson Orin has **2 Deep Learning Accelerator** (DLA) cores that can run inference independently of the GPU. YOLO and MediaPipe models can be compiled to DLA via TensorRT, freeing the GPU entirely for vLLM. This is the key to running vision + LLM simultaneously without contention.
- **C++ inference** — Running STT (whisper.cpp) and TTS natively instead of through Python wrappers would dramatically reduce memory footprint.

This would be a significant engineering effort but would make a truly self-contained Jetson deployment viable.

## Troubleshooting

### General
- **"Connection refused" from LLM** — Make sure your Ollama (or other LLM server) is running and the URL is correct (check `.env` or the settings UI).
- **Slow first response** — The app pre-warms the LLM on startup. If using Ollama, the first model load can be slow; subsequent requests are fast.
- **STT too slow or inaccurate** — Try a different STT model. `tiny.en` is fastest, `medium.en` is most accurate, `small.en` is a good balance.
- **MediaPipe/YOLO import errors** — MediaPipe is included in the base install. For YOLO, install the extra: `uv sync --extra yolo_vision`.
- **Settings not taking effect** — In headless mode, settings are applied when you click Start. If already running, stop and start again.
- **Robot repeats itself / echo loop** — The microphone is picking up the robot's own TTS output. The app has built-in echo suppression (VAD is muted during and 3 seconds after each response), but if your speaker is very loud or close to the mic, try reducing the volume or increasing the distance between them.

### Microphone / Audio
- **No speech detected (silence)** — This is platform-dependent. See [Audio Architecture](#audio-architecture) for details:
  - **On macOS**: Use the microphone selector in the settings dashboard to pick your Mac's built-in mic instead of "Reachy Mini Audio". The USB device returns silence on macOS when the daemon also has it open.
  - **On Linux/Jetson**: The SDK audio path should work by default. If not, check `MIC_GAIN`.
  - **Standalone mode**: Start the daemon with `--deactivate-audio` so the app has exclusive access to the USB mic.
- **No audio input with correct mic selected** — Check `MIC_GAIN` in `.env` or the settings dashboard. On some systems you may need to increase it (e.g., `2.0` or `3.0`).
- **"Audio input buffer overflowed" spam in logs** — This happens when using a direct mic (via the selector) while the SDK recording is also active. The app handles this automatically — if you see it, it's harmless but indicates the SDK recording stream is being drained by nobody. Restarting should clear it.
- **Test microphone button shows "no signal"** — The selected device is returning silence. Try a different device in the dropdown, or check system audio permissions.

### Jetson Orin (vLLM server)
- **vLLM container won't start** — Check that NVIDIA runtime is installed (`docker info | grep nvidia`) and that MAXN power mode is set (`sudo nvpmodel -m 0`).
- **Out of memory** — vLLM needs 8-10GB for a 4B model. Stop other containers and clear caches: `docker system prune -f && sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`.
- **Slow inference** — Verify power mode with `jtop`. If "15W" or similar is shown instead of "MAXN", run `sudo nvpmodel -m 0 && sudo jetson_clocks`. Use quantized models (W4A16/AWQ-4bit) — FP16 is too slow on Jetson's bandwidth.
- **Model not found** — After starting vLLM, verify with `curl http://localhost:30000/v1/models`. The `--served-model-name` flag determines the model name the app should use.
- **SSH tunnel disconnects** — Use `ssh -fNL 30000:localhost:30000 user@jetson-ip` to run the tunnel in the background, or use `autossh` for automatic reconnection.

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
