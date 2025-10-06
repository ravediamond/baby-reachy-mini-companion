# Reachy Mini conversation demo

Working repo, we should turn this into a ReachyMini app at some point maybe ?

## Installation
You can set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12.1  # Create a virtual environment with Python 3.12.1
source .venv/bin/activate
uv sync
```
> Note: The `pyproject.toml` expects `reachy-mini-dances-library` to be located in the same directory as this project.

## Run

```bash
reachy-mini-conversation-demo
```

## Runtime Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--sim` | *(flag)* | off | Run in **simulation mode** (no physical robot required). |
| `--vision` | *(flag)* | off | Enable the **vision system** (must be paired with `--vision-provider`). |
| `--vision-provider` | `local`, `openai` | `local` | Select vision backend:<br>• **local** → Hugging Face VLM (SmolVLM2) runs on your machine.<br>• **openai** → OpenAI multimodal models via API (requires `OPENAI_API_KEY`). |
| `--head-tracking` | *(flag)* | off | Enable **head tracking** (ignored when `--sim` is active). |
| `--debug` | *(flag)* | off | Enable **debug logging** (default log level is INFO). |

## Examples
- Simulated run with OpenAI Vision:
```
reachy-mini-conversation-demo --sim --vision --vision-provider=openai
```
