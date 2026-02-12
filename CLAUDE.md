# Baby Reachy-Mini Companion - Project Context

## What This Is

A **baby nursery companion** built on the Reachy Mini robot by Pollen Robotics. The project is a fully local AI stack — 7+ models orchestrated on-device with zero cloud dependency. It's entered in the **NVIDIA/HuggingFace GTC Golden Ticket contest** in two categories: **Reachy Mini usage** and **Ollama**.

The developer is a new dad who built this for his actual baby. His broader goal is to transition into robotics focused on helping people (inspired by Enchanted Tools' hospital companion robots). This is his first step toward accessible assistive robotics.

## Contest Context

- **Contest**: NVIDIA GTC Golden Ticket — 9 winners, each in a category
- **Categories entered**: Reachy Mini usage, Ollama
- **Key differentiators vs other contestants**:
  - The ONLY submission running fully local (no cloud APIs)
  - The ONLY one deploying on NVIDIA Jetson Orin
  - The ONLY one using local SLMs instead of cloud LLMs
  - Was #1 in community rankings
- **Judging criteria** (equal weight, 1-10 each): Technical Innovation, Effective Use of Technology, Potential Impact, Presentation Quality

## Architecture

### Processing Pipeline
```
Audio → Silero VAD → Faster-Whisper STT → Ollama LLM (with tool calling) → Kokoro TTS → Speaker
                                              ↓
                              Tool dispatch (16+ tools)
                              ↓           ↓           ↓
                          Camera/VLM   Signal alerts   Robot motion
```

### Autonomous Detection Loops (run in background)
- **Audio classifier** (YAMNet): Detects baby cries → sends Signal photo alert directly (bypasses LLM) + injects system event into LLM → LLM calls soothe_baby
- **Danger detector** (YOLO): Scans camera every 2s for hazardous objects (scissors, knives, forks) → sends Signal photo alert directly (bypasses LLM) + injects system event → LLM speaks warning
- **Signal poller**: Receives remote text messages → processed through LLM → responds via Signal

**Critical design rule**: Safety-critical notifications (cry alerts, danger alerts) are sent directly in handler code via `_send_cry_photo_alert()` / `_send_danger_photo_alert()`. They do NOT depend on the LLM calling tools — SLMs (3B–4B) can't reliably chain 3+ sequential tool calls.

### Movement System
- 100 Hz control loop with monotonic clock phase alignment
- Primary moves (dances, emotions) are mutually exclusive
- Secondary moves (speech sway via HeadWobbler) are additive offsets
- HeadWobbler: converts TTS audio stream into head movement for natural speech appearance

## Repository Structure

```
src/reachy_mini_conversation_app/
├── main.py                  # Entrypoint, wires all components together
├── console.py               # Headless mode with settings dashboard (FastAPI)
├── config.py                # Configuration from .env, includes feature flags
├── moves.py                 # 100Hz movement control loop
├── camera_worker.py         # 30Hz camera polling with frame buffering
├── prompts.py               # Dynamic prompt loading from profiles
├── local/
│   ├── handler.py           # Core pipeline: VAD→STT→LLM→TTS + system events
│   ├── llm.py               # OpenAI-compatible LLM client with streaming + tools
│   ├── stt.py               # Faster-Whisper wrapper
│   ├── tts.py               # Kokoro ONNX wrapper
│   └── vad.py               # Silero VAD wrapper
├── tools/
│   ├── core_tools.py        # Tool base class, registry, ToolDependencies dataclass
│   ├── camera.py            # VLM visual question answering
│   ├── soothe_baby.py       # Rocking motions + lullaby
│   ├── check_baby_crying.py # Query audio classifier status
│   ├── check_danger.py      # Query YOLO danger detector status
│   ├── send_signal.py       # Send text via Signal
│   ├── send_signal_photo.py # Capture frame + send via Signal
│   ├── dance.py             # Movement primitives
│   ├── story_time.py        # Children's story narration
│   ├── move_head.py         # Direct head control
│   └── ...
├── vision/
│   ├── processors.py        # VisionProcessor (API-based VLM) + VisionManager
│   └── danger_detector.py   # YOLO object detection for baby safety
├── audio/
│   ├── classifier.py        # YAMNet ONNX audio event classifier
│   ├── head_wobbler.py      # Audio-reactive head movement during speech
│   └── speech_tapper.py     # Loudness/VAD processing
├── input/
│   └── signal_interface.py  # Signal-CLI bridge
├── profiles/
│   └── default/
│       ├── instructions.txt # System prompt
│       └── tools.txt        # Enabled tools list
├── static/                  # Settings dashboard (HTML/CSS/JS)
│   ├── index.html           # Dashboard with LLM settings + feature toggles
│   ├── style.css
│   └── main.js
└── images/                  # Avatar images for Gradio chatbot
```

## Key Design Patterns

### System Events (autonomous action pattern)
Used by both audio classifier and danger detector:
1. Background task detects something (cry, dangerous object)
2. **Direct alert**: Handler sends Signal photo alert immediately (guaranteed delivery)
3. **LLM event**: Calls `_process_system_event(text)` on the handler
4. Injects `[System Notification: ...]` as a user message into the LLM
5. LLM runs a tool-calling loop (max 5 turns) and decides what to do (soothe, speak warning)
6. Throttled to prevent spam (10s for cries, 30s for danger)

### SLM Tool-Calling Limitations
SLMs (3B–4B) reliably call 1–2 tools per turn but **fail at 3+ sequential tool calls**. Each tool call consumes one LLM turn (generate call → execute → return result → next turn). With a `max_turns` limit, 3+ calls leave no room for a spoken response. Even with higher limits, the model often generates text instead of making the next tool call.

**Rule**: Never gate safety-critical actions on LLM tool-calling behavior. Use the LLM for reasoning (what to say, whether to soothe) but send notifications/alerts deterministically in code.

### Ollama Streaming Tool Call Bug
Ollama sends all tool calls under stream index 0, unlike OpenAI which uses distinct indices. The `llm.py` streaming parser uses a `_stream_to_buf` mapping to detect when a new tool name arrives on an already-populated buffer index and assigns a synthetic index. Without this fix, multiple tool calls from Ollama get concatenated into one garbled call.

### ToolDependencies
Central dataclass injected into all tools. Contains: `reachy_mini`, `movement_manager`, `camera_worker`, `vision_manager`, `head_wobbler`, `audio_classifier_status`, `vision_threat_status`, `speak_func`, `motion_duration_s`.

### Feature Flags
Five toggleable features in `config.py`, controllable via `.env` or the settings dashboard:
- `FEATURE_CRY_DETECTION` — YAMNet audio classifier
- `FEATURE_AUTO_SOOTHE` — soothe_baby + check_baby_crying tools
- `FEATURE_DANGER_DETECTION` — YOLO visual safety scanner + check_danger tool
- `FEATURE_STORY_TIME` — story_time tool
- `FEATURE_SIGNAL_ALERTS` — Signal messaging tools

Feature flags work by: (1) excluding tools from `tool_specs` via `get_tool_specs(exclusion_list=...)`, and (2) conditionally loading background services in `handler.start_up()`.

### Profile System
Profiles live in `src/.../profiles/<name>/`. Each has `instructions.txt` (system prompt) and `tools.txt` (enabled tools, one per line). Tools are loaded dynamically via `importlib` — profile-local tools take priority over shared tools.

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Ollama / vLLM (OpenAI-compatible) | Local language model with tool calling |
| STT | Faster-Whisper | Real-time speech-to-text |
| TTS | Kokoro (ONNX) | Neural speech synthesis |
| VAD | Silero VAD (PyTorch) | Voice activity detection |
| Vision | Qwen2.5-VL / Qwen3-VL (via Ollama) | Visual question answering |
| Object Detection | YOLO v26 nano (general COCO model) | Danger detection |
| Audio Classification | YAMNet (ONNX) | Baby cry / environmental sound detection |
| Messaging | signal-cli | Remote text + photo alerts |
| Robot SDK | reachy-mini SDK (Zenoh) | Robot control |
| Web UI | Gradio 5.50 / FastAPI | Dashboard + audio streaming |
| Audio Streaming | FastRTC | Bidirectional audio |

### Tested LLM Models
- `ministral-3:3b` — **recommended default**, best reasoning at 3B. Does NOT work on vLLM v0.14 (Jetson). Note: the model name has a hyphen (`ministral-3:3b`), not a colon.
- `qwen2.5:3b` — stable everywhere, good fallback
- `qwen2.5-vl:3b` — vision model, stable on Ollama and vLLM
- `qwen3-vl:4b` — better vision, but dynamic patching requires latest engine versions

## Deployment Scenarios

1. **Everything on Mac** — Ollama runs LLM locally, all audio on Mac
2. **Hybrid: Mac + Jetson** — App/audio on Mac, LLM on Jetson via SSH tunnel
3. **Everything on Jetson Orin** — Full stack on edge device using NVIDIA's official container (`ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`)

## Settings Dashboard

The headless settings UI (`static/index.html`) has three panels (Features and Personality are collapsible):
1. **LLM Settings** — Server URL, model, API key, STT model size. Includes a hint about needing Ollama/vLLM running externally.
2. **Features** (collapsible) — 5 toggle switches (cry detection, auto soothe, danger detection, story time, Signal alerts). Signal toggle reveals a phone number input.
3. **Personality Studio** (collapsible, starts collapsed) — Profile selection, instructions editor, tools checkboxes, voice selection.

The **Start button** is positioned below the Features panel. Settings are persisted to the instance `.env` file and sent to the handler when the user clicks "Start".

## Presentation Assets

- `docs/assets/baby-reachy-mini.jpg` — Photo of Reachy Mini among baby toys (hero image)
- `docs/assets/reachy-demo.mp4` — 33MB demo video (embedded in README)
- `docs/assets/conversation_app_arch.svg` — Architecture diagram (embedded in README)
- `app.py` — Gradio landing page with mission statement, design principles, differentiators, and feature cards

## Landing Page Structure (app.py)

Flow: Hero → Deploy tags → **Mission statement** (personal story) → **Design principles** (privacy, consumer hardware, physical safety, empathy) → **What makes this different** (local, 7+ models, safety monitor, Jetson) → **Capabilities** (6 feature cards) → Getting started → Footer.

## Development

```bash
# Install (base includes full voice pipeline + YOLO vision)
uv sync

# With wireless Reachy Mini support
uv sync --extra local

# Configure
cp .env.example .env

# Run (Gradio UI)
uv run reachy-mini-conversation-app --gradio

# Run (headless with dashboard)
uv run reachy-mini-conversation-app

# Quality checks
ruff format .
ruff check . --fix
mypy --strict .
pytest tests/ -v
```

## Git & Deployment

- **Main branch**: `develop` (used for PRs)
- **Working branch**: `main` — contest submission and HuggingFace Space are deployed from `main`
- **HuggingFace mirror**: `git push hf main:main` (to `ravediamond/baby-reachy-mini-companion`)
- **CI**: GitHub Actions — lint, typecheck, tests, uv lock check, semantic release, HF Space sync
- **Push workflow**: After committing, push to both `origin main` and `hf main:main`

## Design Philosophy

- **Privacy is non-negotiable** — all processing local, no cloud, no data leaves the device
- **Consumer hardware enables adoption** — runs on $700 Jetson Orin NX or existing Mac
- **Physical safety by design** — Reachy Mini has no hands/manipulators
- **Empathy is the key to acceptance** — a robot that ignores human distress has failed its purpose
- **AI companions for children should be explored openly** — better to build transparent, parent-controlled experiences than wait for closed commercial products
- **Never gate safety on model behavior** — notifications and alerts are deterministic code, not LLM tool calls

## Known Gotchas

- **Kokoro TTS can't handle onomatopoeia** — "Shhh" gets spelled out as "s s s h". Use real words like "Hush now" instead.
- **YOLO danger detection uses multi-frame confirmation** — `yolo26s` (small) model with 3-of-5 frame confirmation to filter hallucinations. A dangerous object must appear in 3 of the last 5 frames before triggering an alert. Confidence threshold is 0.2.
- **Echo suppression timing** — The cooldown must be relative to estimated playback end, not pipeline completion. Pipeline finishes when TTS audio is *queued*, not when it finishes *playing*. `_playback_end_mono` tracks the estimated speaker finish time.
- **Ollama tool call indices** — Ollama sends all streaming tool calls under index 0 (unlike OpenAI). The `llm.py` parser handles this with synthetic buffer indices.
- **signal-cli path** — Not always on PATH. `signal_interface.py` checks `/opt/homebrew/bin/signal-cli` and `/usr/local/bin/signal-cli` as fallbacks.
- **Feature descriptions in docs** — The robot speaks calming words (not sings lullabies). Stories are a fixed list (Three Little Pigs, Goldilocks), not improvised. Keep docs accurate.
