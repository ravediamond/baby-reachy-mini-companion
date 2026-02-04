# Reachy Mini Conversation App - Gemini Context

This repository contains the **Reachy Mini Conversation App**, a sophisticated conversational assistant designed for the Reachy Mini robot. It focuses on privacy, low latency, and multimodal interaction by leveraging local AI models.

## Core Goals

- **Omni-Channel Interaction**: The assistant is designed to be accessible both locally and remotely. It can "talk" via:
    - **Voice**: Real-time audio interaction using local Speech-to-Text (STT) and Text-to-Speech (TTS).
    - **Signal**: Remote text-based interaction via the Signal messaging protocol.
- **Multimodal Perception & Response**:
    - **Vision**: The assistant can perceive its environment using camera input and vision models (e.g., SmolVLM2), allowing it to incorporate visual context into conversations.
    - **Motion**: Beyond speaking, the assistant can respond by performing dances, expressing emotions, and tracking faces.
- **Intelligent Tool-Use**: The assistant uses an LLM (e.g., Qwen 2.5 via Ollama) to intelligently decide when to use specific tools to interact with the physical world or perform tasks.
- **Fully Local Execution**: Designed to run entirely on-device (Mac M-series or NVIDIA Jetson) to ensure user privacy and offline capability.

## Repository Structure

- `src/reachy_mini_conversation_app/`: Main source code.
    - `local/`: Implementation of local AI services (LLM, STT, TTS, VAD).
    - `input/`: Input interface implementations, including the Signal bridge.
    - `tools/`: A suite of tools for controlling the robot (head movement, dances, emotions).
    - `vision/`: Logic for vision processing and head tracking.
    - `profiles/`: Configuration for different assistant personalities and enabled toolsets.
    - `audio/`: Specialized audio utilities like the "Head Wobbler" for lip-sync-like movement.
- `scripts/`: Utility scripts for environment setup and testing.
- `docs/`: Architectural diagrams and detailed documentation.
- `tests/`: Unit and integration tests for the various components.

## Key Technologies

- **LLM**: Ollama (OpenAI-compatible local server).
- **STT**: Faster-Whisper for high-performance transcription.
- **TTS**: Kokoro (ONNX) for high-quality neural voice synthesis.
- **VAD**: Silero VAD for robust voice activity detection.
- **Messaging**: Signal-CLI for remote communication.
- **Robot Interface**: Reachy Mini SDK and Zenoh.
