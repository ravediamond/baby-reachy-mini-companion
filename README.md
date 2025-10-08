# Reachy Mini conversation demo

Conversational demo for the Reachy Mini robot combining OpenAI's realtime APIs, vision pipelines, and choreographed motion libraries.

## Overview
- Real-time audio conversation loop powered by the OpenAI realtime API and `fastrtc` for low-latency streaming.
- Motion control queue that blends scripted dances, recorded emotions, idle breathing, and speech-reactive head wobbling.
- Optional camera worker with YOLO or MediaPipe-based head tracking and LLM-accessible scene capture.

## Features
- Async tool dispatch integrates robot motion, camera capture, and optional facial recognition helpers.
- Gradio web UI provides audio chat and transcript display.
- Movement manager keeps real-time control in a dedicated thread with safeguards against abrupt pose changes.

## Installation

### Using uv
You can set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12.1  # Create a virtual environment with Python 3.12.1
source .venv/bin/activate
uv sync
```

To include optional vision dependencies:
```
uv sync --extra local_vision        # For local PyTorch/Transformers vision
uv sync --extra yolo_vision         # For YOLO-based vision
uv sync --extra mediapipe_vision    # For MediaPipe-based vision
uv sync --extra all_vision          # For all vision features
```

You can combine extras or include dev dependencies:
```
uv sync --extra all_vision --group dev
```

### Using pip (test on Ubuntu 24.04)

```bash
python -m venv .venv # Create a virtual environment
source .venv/bin/activate
pip install -e .
```

Install optional extras depending on the feature set you need:

```bash
# Vision stacks (choose at least one if you plan to run head tracking)
pip install -e .[local_vision]
pip install -e .[yolo_vision]
pip install -e .[mediapipe_vision]
pip install -e .[all_vision]        # installs every vision extra

# Tooling for development workflows
pip install -e .[dev]
```

Some wheels (e.g. PyTorch) are large and require compatible CUDA or CPU builds. Expect the `local_vision` extra to take significantly more disk space than YOLO or MediaPipe.

## Optional dependency groups

| Extra | Purpose | Notes |
|-------|---------|-------|
| `local_vision` | Run the local VLM (SmolVLM2) through PyTorch/Transformers. | GPU recommended; installs large packages (~2 GB).
| `yolo_vision` | YOLOv8 tracking via `ultralytics` and `supervision`. | CPU friendly; supports the `--head-tracker yolo` option.
| `mediapipe_vision` | Lightweight landmark tracking with MediaPipe. | Works on CPU; enables `--head-tracker mediapipe`.
| `all_vision` | Convenience alias installing every vision extra. | Only use if you need to experiment with all providers.
| `dev` | Developer tooling (`pytest`, `ruff`). | Add on top of either base or `all_vision` environments.

## Configuration

1. Copy `.env.example` to `.env`.
2. Fill in the required values, notably the OpenAI API key.

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required. Grants access to the OpenAI realtime endpoint.
| `MODEL_NAME` | Override the realtime model (defaults to `gpt-realtime`).
| `HF_HOME` | Cache directory for local Hugging Face downloads.
| `HF_TOKEN` | Optional token for Hugging Face models.

## Running the demo

Activate your virtual environment, ensure the Reachy Mini robot (or simulator) is reachable, then launch:

```bash
reachy-mini-conversation-demo
```

The app starts a Gradio UI served locally (http://127.0.0.1:7860/). When running on a headless host, use `--headless`.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--head-tracker {yolo,mediapipe}` | `None` | Select a head-tracking backend when a camera is available. Requires the matching optional extra. |
| `--no-camera` | `False` | Run without camera capture or head tracking. |
| `--headless` | `False` | Suppress launching the Gradio UI (useful on remote machines). |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |

### Examples
- Run on hardware with MediaPipe head tracking:

  ```bash
  reachy-mini-conversation-demo --head-tracker mediapipe
  ```

- Disable the camera pipeline (audio-only conversation):

  ```bash
  reachy-mini-conversation-demo --no-camera
  ```

## LLM tools exposed to the assistant

| Tool | Action | Dependencies |
|------|--------|--------------|
| `move_head` | Queue a head pose change (left/right/up/down/front). | Core install only. |
| `camera` | Capture the latest camera frame and optionally query a vision backend. | Requires camera worker; vision analysis depends on selected extras. |
| `head_tracking` | Enable or disable face-tracking offsets. | Camera worker with configured head tracker. |
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Core install only. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face assets. | Needs `HF_TOKEN` for the recorded emotions dataset. |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `get_person_name` | Attempt DeepFace-based recognition of the current person. | Disabled by default (`ENABLE_FACE_RECOGNITION=False`); requires `deepface` and a local face database. |
| `do_nothing` | Explicitly remain idle. | Core install only. |

## Development workflow
- Install the dev group extras: `uv sync --group dev` or `pip install -e .[dev]`.
- Run formatting and linting: `ruff check .`.
- Execute the test suite: `pytest`.
- When iterating on robot motions, keep the control loop responsive—offload blocking work using the helpers in `tools.py`.


## License

Apache 2.0
