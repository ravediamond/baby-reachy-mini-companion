"""Bidirectional local audio stream with optional settings UI.

In headless mode, there is no Gradio UI. We expose a minimal settings page
via the Reachy Mini Apps settings server to let non-technical users configure
the app — either the OpenAI API key (realtime mode) or the local LLM settings
(local mode: URL, model, API key, STT model).

The settings UI is served from this package's ``static/`` folder.
Once set, values are persisted to the app instance's ``.env`` file
(if available) and used immediately.
"""

import os
import sys
import time
import asyncio
import logging
import threading
from typing import Any, List, Optional
from pathlib import Path

from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes


try:
    # FastAPI is provided by the Reachy Mini Apps runtime
    from fastapi import FastAPI, Request, Response
    from pydantic import BaseModel
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.staticfiles import StaticFiles
except Exception:  # pragma: no cover - only loaded when settings_app is used
    FastAPI = object  # type: ignore[misc,assignment]
    Request = object  # type: ignore[misc,assignment]
    FileResponse = object  # type: ignore[misc,assignment]
    JSONResponse = object  # type: ignore[misc,assignment]
    StaticFiles = object  # type: ignore[misc,assignment]
    BaseModel = object  # type: ignore[misc,assignment]


logger = logging.getLogger(__name__)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: Any,
        robot: ReachyMini,
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
    ):
        """Initialize the stream with an OpenAI realtime handler and pipelines.

        - ``settings_app``: the Reachy Mini Apps FastAPI to attach settings endpoints.
        - ``instance_path``: directory where per-instance ``.env`` should be stored.
        """
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        # Allow the handler to flush the player queue when appropriate.
        self.handler._clear_queue = self.clear_audio_queue
        self._settings_app: Optional[FastAPI] = settings_app
        self._instance_path: Optional[str] = instance_path
        self._settings_initialized = False
        self._asyncio_loop = None
        # Gate for local mode: blocks launch() until user clicks "Start" in the UI
        self._start_event = threading.Event()
        self._pipeline_started = False

        # Register dashboard routes immediately so the UI is available
        # while the robot and pipeline are still initializing.
        self._init_settings_ui_if_needed()

    # ---- Settings UI (only when API key is missing) ----
    def _read_env_lines(self, env_path: Path) -> list[str]:
        """Load env file contents or a template as a list of lines."""
        inst = env_path.parent
        try:
            if env_path.exists():
                try:
                    return env_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    return []
            template_text = None
            ex = inst / ".env.example"
            if ex.exists():
                try:
                    template_text = ex.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                try:
                    cwd_example = Path.cwd() / ".env.example"
                    if cwd_example.exists():
                        template_text = cwd_example.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                packaged = Path(__file__).parent / ".env.example"
                if packaged.exists():
                    try:
                        template_text = packaged.read_text(encoding="utf-8")
                    except Exception:
                        template_text = None
            return template_text.splitlines() if template_text else []
        except Exception:
            return []

    def _persist_api_key(self, key: str) -> None:
        """Persist API key to environment and instance ``.env`` if possible.

        Behavior:
        - Always sets ``OPENAI_API_KEY`` in process env and in-memory config.
        - Writes/updates ``<instance_path>/.env``:
          * If ``.env`` exists, replaces/append OPENAI_API_KEY line.
          * Else, copies template from ``<instance_path>/.env.example`` when present,
            otherwise falls back to the packaged template
            ``reachy_mini_conversation_app/.env.example``.
          * Ensures the resulting file contains the full template plus the key.
        - Loads the written ``.env`` into the current process environment.
        """
        k = (key or "").strip()
        if not k:
            return
        # Update live process env and config so consumers see it immediately
        try:
            os.environ["OPENAI_API_KEY"] = k
        except Exception:  # best-effort
            pass
        try:
            config.OPENAI_API_KEY = k
        except Exception:
            pass

        if not self._instance_path:
            return
        try:
            inst = Path(self._instance_path)
            env_path = inst / ".env"
            lines = self._read_env_lines(env_path)
            replaced = False
            for i, ln in enumerate(lines):
                if ln.strip().startswith("OPENAI_API_KEY="):
                    lines[i] = f"OPENAI_API_KEY={k}"
                    replaced = True
                    break
            if not replaced:
                lines.append(f"OPENAI_API_KEY={k}")
            final_text = "\n".join(lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted OPENAI_API_KEY to %s", env_path)

            # Load the newly written .env into this process to ensure downstream imports see it
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist OPENAI_API_KEY: %s", e)

    def _persist_personality(self, profile: Optional[str]) -> None:
        """Persist the startup personality to the instance .env and config."""
        selection = (profile or "").strip() or None
        try:
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(selection)
        except Exception:
            pass

        if not self._instance_path:
            return
        try:
            env_path = Path(self._instance_path) / ".env"
            lines = self._read_env_lines(env_path)
            replaced = False
            for i, ln in enumerate(list(lines)):
                if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                    if selection:
                        lines[i] = f"REACHY_MINI_CUSTOM_PROFILE={selection}"
                    else:
                        lines.pop(i)
                    replaced = True
                    break
            if selection and not replaced:
                lines.append(f"REACHY_MINI_CUSTOM_PROFILE={selection}")
            if selection is None and not env_path.exists():
                return
            final_text = "\n".join(lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted startup personality to %s", env_path)
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist REACHY_MINI_CUSTOM_PROFILE: %s", e)

    def _read_persisted_personality(self) -> Optional[str]:
        """Read persisted startup personality from instance .env (if any)."""
        if not self._instance_path:
            return None
        env_path = Path(self._instance_path) / ".env"
        try:
            if env_path.exists():
                for ln in env_path.read_text(encoding="utf-8").splitlines():
                    if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                        _, _, val = ln.partition("=")
                        v = val.strip()
                        return v or None
        except Exception:
            pass
        return None

    def _is_local_handler(self) -> bool:
        """Check whether the handler is the local LLM handler."""
        try:
            from reachy_mini_conversation_app.local.handler import LocalSessionHandler

            return isinstance(self.handler, LocalSessionHandler)
        except Exception:
            return False

    _FEATURE_KEYS = (
        "FEATURE_CRY_DETECTION",
        "FEATURE_AUTO_SOOTHE",
        "FEATURE_DANGER_DETECTION",
        "FEATURE_STORY_TIME",
        "FEATURE_SIGNAL_ALERTS",
        "FEATURE_HEAD_TRACKING",
    )

    def _persist_local_llm_settings(self, settings: dict[str, str]) -> None:
        """Persist local LLM and feature settings to environment, config, and instance .env.

        Accepted keys: LOCAL_LLM_URL, LOCAL_LLM_MODEL, LOCAL_LLM_API_KEY, LOCAL_STT_MODEL,
        SIGNAL_USER_PHONE, and all FEATURE_* flags.
        """
        env_keys = (
            "LOCAL_LLM_URL", "LOCAL_LLM_MODEL", "LOCAL_LLM_API_KEY", "LOCAL_STT_MODEL",
            "SIGNAL_USER_PHONE", "MIC_GAIN",
            *self._FEATURE_KEYS,
        )
        for key in env_keys:
            val = (settings.get(key) or "").strip()
            if not val:
                continue
            try:
                os.environ[key] = val
            except Exception:
                pass
            try:
                # Feature flags are booleans on config
                if key.startswith("FEATURE_"):
                    setattr(config, key, val.lower() == "true")
                elif key == "MIC_GAIN":
                    setattr(config, key, float(val))
                else:
                    setattr(config, key, val)
            except Exception:
                pass

        if not self._instance_path:
            return
        try:
            inst = Path(self._instance_path)
            env_path = inst / ".env"
            lines = self._read_env_lines(env_path)
            for key in env_keys:
                val = (settings.get(key) or "").strip()
                if not val:
                    continue
                replaced = False
                for i, ln in enumerate(lines):
                    if ln.strip().startswith(f"{key}="):
                        lines[i] = f'{key}="{val}"'
                        replaced = True
                        break
                if not replaced:
                    lines.append(f'{key}="{val}"')
            final_text = "\n".join(lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted local LLM settings to %s", env_path)
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist local LLM settings: %s", e)

    def _init_settings_ui_if_needed(self) -> None:
        """Attach minimal settings UI to the settings app.

        Always mounts the UI when a settings_app is provided so that users
        see a confirmation message even if the API key is already configured.
        """
        if self._settings_initialized:
            return
        if self._settings_app is None:
            return

        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"

        if hasattr(self._settings_app, "mount"):
            try:
                # Serve /static/* assets
                self._settings_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            except Exception:
                pass

        class ApiKeyPayload(BaseModel):
            openai_api_key: str

        # GET / -> index.html
        @self._settings_app.get("/")
        def _root() -> FileResponse:
            return FileResponse(str(index_file))

        # GET /favicon.ico -> optional, avoid noisy 404s on some browsers
        @self._settings_app.get("/favicon.ico")
        def _favicon() -> Response:
            return Response(status_code=204)

        # GET /status -> whether key is set
        @self._settings_app.get("/status")
        def _status() -> JSONResponse:
            has_key = bool(config.OPENAI_API_KEY and str(config.OPENAI_API_KEY).strip())
            return JSONResponse({"has_key": has_key})

        # GET /ready -> whether backend finished loading tools
        @self._settings_app.get("/ready")
        def _ready() -> JSONResponse:
            try:
                mod = sys.modules.get("reachy_mini_conversation_app.tools.core_tools")
                ready = bool(getattr(mod, "_TOOLS_INITIALIZED", False)) if mod else False
            except Exception:
                ready = False
            return JSONResponse({"ready": ready})

        # POST /openai_api_key -> set/persist key
        @self._settings_app.post("/openai_api_key")
        def _set_key(payload: ApiKeyPayload) -> JSONResponse:
            key = (payload.openai_api_key or "").strip()
            if not key:
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)
            self._persist_api_key(key)
            return JSONResponse({"ok": True})

        # POST /validate_api_key -> validate key without persisting it
        @self._settings_app.post("/validate_api_key")
        async def _validate_key(payload: ApiKeyPayload) -> JSONResponse:
            key = (payload.openai_api_key or "").strip()
            if not key:
                return JSONResponse({"valid": False, "error": "empty_key"}, status_code=400)

            # Try to validate by checking if we can fetch the models
            try:
                import httpx

                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get("https://api.openai.com/v1/models", headers=headers)
                    if response.status_code == 200:
                        return JSONResponse({"valid": True})
                    elif response.status_code == 401:
                        return JSONResponse({"valid": False, "error": "invalid_api_key"}, status_code=401)
                    else:
                        return JSONResponse(
                            {"valid": False, "error": "validation_failed"}, status_code=response.status_code
                        )
            except Exception as e:
                logger.warning(f"API key validation failed: {e}")
                return JSONResponse({"valid": False, "error": "validation_error"}, status_code=500)

        # GET /app_mode -> whether we're running local or openai-realtime
        @self._settings_app.get("/app_mode")
        def _app_mode() -> JSONResponse:
            return JSONResponse({"mode": "local" if self._is_local_handler() else "openai"})

        # GET /app_state -> configuring or running
        @self._settings_app.get("/app_state")
        def _app_state() -> JSONResponse:
            return JSONResponse({
                "state": "running" if self._pipeline_started else "configuring",
            })

        # GET /local_llm_settings -> current local LLM configuration
        @self._settings_app.get("/local_llm_settings")
        def _get_local_llm_settings() -> JSONResponse:
            return JSONResponse({
                "LOCAL_LLM_URL": config.LOCAL_LLM_URL or "",
                "LOCAL_LLM_MODEL": config.LOCAL_LLM_MODEL or "",
                "LOCAL_LLM_API_KEY": config.LOCAL_LLM_API_KEY or "",
                "LOCAL_STT_MODEL": config.LOCAL_STT_MODEL or "",
            })

        # GET /feature_settings -> current feature flags
        @self._settings_app.get("/feature_settings")
        def _get_feature_settings() -> JSONResponse:
            return JSONResponse({
                "FEATURE_CRY_DETECTION": config.FEATURE_CRY_DETECTION,
                "FEATURE_AUTO_SOOTHE": config.FEATURE_AUTO_SOOTHE,
                "FEATURE_DANGER_DETECTION": config.FEATURE_DANGER_DETECTION,
                "FEATURE_STORY_TIME": config.FEATURE_STORY_TIME,
                "FEATURE_SIGNAL_ALERTS": config.FEATURE_SIGNAL_ALERTS,
                "FEATURE_HEAD_TRACKING": config.FEATURE_HEAD_TRACKING,
                "SIGNAL_USER_PHONE": config.SIGNAL_USER_PHONE or "",
                "MIC_GAIN": config.MIC_GAIN,
            })

        # POST /test_mic -> record a short audio clip and check signal level
        @self._settings_app.post("/test_mic")
        async def _test_mic(request: Request) -> JSONResponse:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            gain = float(raw.get("MIC_GAIN", 1.0))
            try:
                import sounddevice as sd
                duration = 1.5  # seconds
                sr = 16000
                recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
                sd.wait()
                audio = recording.flatten() * gain
                rms = float((audio ** 2).mean() ** 0.5)
                peak = float(abs(audio).max())
                # Thresholds tuned for speech detection
                if peak < 0.005:
                    verdict = "no_signal"
                    msg = "No audio signal detected. Check that your microphone is connected and not muted."
                elif rms < 0.01:
                    verdict = "too_quiet"
                    msg = f"Audio detected but very quiet (RMS: {rms:.4f}). Try increasing the mic gain or moving closer."
                else:
                    verdict = "ok"
                    msg = f"Microphone working (RMS: {rms:.4f}, Peak: {peak:.4f})."
                return JSONResponse({"ok": True, "verdict": verdict, "message": msg, "rms": round(rms, 5), "peak": round(peak, 5)})
            except Exception as e:
                return JSONResponse({"ok": False, "verdict": "error", "message": f"Mic test failed: {e}"}, status_code=500)

        # POST /test_llm -> check if the LLM endpoint is reachable
        @self._settings_app.post("/test_llm")
        async def _test_llm(request: Request) -> JSONResponse:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            url = (raw.get("LOCAL_LLM_URL") or config.LOCAL_LLM_URL or "").strip().rstrip("/")
            model = (raw.get("LOCAL_LLM_MODEL") or config.LOCAL_LLM_MODEL or "").strip()
            api_key = (raw.get("LOCAL_LLM_API_KEY") or config.LOCAL_LLM_API_KEY or "").strip()
            if not url:
                return JSONResponse({"ok": False, "verdict": "no_url", "message": "No server URL configured."})
            try:
                import httpx
                headers = {}
                if api_key and api_key != "ollama":
                    headers["Authorization"] = f"Bearer {api_key}"
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{url}/models", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        models = [m.get("id", "") for m in data.get("data", [])]
                        if model and model in models:
                            return JSONResponse({"ok": True, "verdict": "ok", "message": f"Connected. Model '{model}' is available.", "models": models})
                        elif model:
                            return JSONResponse({"ok": True, "verdict": "model_missing", "message": f"Connected but model '{model}' not found. Available: {', '.join(models[:5])}", "models": models})
                        else:
                            return JSONResponse({"ok": True, "verdict": "ok", "message": f"Connected. Available models: {', '.join(models[:5])}", "models": models})
                    else:
                        return JSONResponse({"ok": False, "verdict": "error", "message": f"Server returned {resp.status_code}."})
            except httpx.ConnectError:
                return JSONResponse({"ok": False, "verdict": "unreachable", "message": f"Cannot connect to {url}. Is the server running?"})
            except Exception as e:
                return JSONResponse({"ok": False, "verdict": "error", "message": f"Connection test failed: {e}"})

        # POST /start_app -> save settings and start the pipeline
        @self._settings_app.post("/start_app")
        async def _start_app(request: Request) -> JSONResponse:
            if self._pipeline_started:
                return JSONResponse({"ok": True, "already_running": True})
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            # Persist whatever settings were sent
            if raw:
                self._persist_local_llm_settings(raw)
                # Also update the handler's live LLM URL and model so start_up() uses them
                try:
                    url = (raw.get("LOCAL_LLM_URL") or "").strip()
                    model = (raw.get("LOCAL_LLM_MODEL") or "").strip()
                    if url:
                        self.handler.llm_url = url
                    if model:
                        self.handler.llm_model = model
                except Exception:
                    pass
            # Unblock launch()
            self._start_event.set()
            return JSONResponse({"ok": True})

        self._settings_initialized = True

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops.

        If the OpenAI key is missing, expose a tiny settings UI via the
        Reachy Mini settings server to collect it before starting streams.
        """
        self._stop_event.clear()

        # Try to load an existing instance .env first (covers subsequent runs)
        if self._instance_path:
            try:
                from dotenv import load_dotenv

                from reachy_mini_conversation_app.config import set_custom_profile

                env_path = Path(self._instance_path) / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    # Update config with newly loaded values
                    new_key = os.getenv("OPENAI_API_KEY", "").strip()
                    if new_key:
                        try:
                            config.OPENAI_API_KEY = new_key
                        except Exception:
                            pass
                    new_profile = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
                    if new_profile is not None:
                        try:
                            set_custom_profile(new_profile.strip() or None)
                        except Exception:
                            pass
                    # Reload local LLM settings from instance .env
                    for env_key in ("LOCAL_LLM_URL", "LOCAL_LLM_MODEL", "LOCAL_LLM_API_KEY", "LOCAL_STT_MODEL"):
                        val = os.getenv(env_key, "").strip()
                        if val:
                            try:
                                setattr(config, env_key, val)
                            except Exception:
                                pass
            except Exception:
                pass

        # If key is still missing, try to download one from HuggingFace
        # ONLY if we are using OpenAI Realtime. Local handler doesn't need it.
        from reachy_mini_conversation_app.local.handler import LocalSessionHandler
        is_local = isinstance(self.handler, LocalSessionHandler)

        if not is_local and not (config.OPENAI_API_KEY and str(config.OPENAI_API_KEY).strip()):
            logger.info("OPENAI_API_KEY not set, attempting to download from HuggingFace...")
            try:
                from gradio_client import Client
                client = Client("HuggingFaceM4/gradium_setup", verbose=False)
                key, status = client.predict(api_name="/claim_b_key")
                if key and key.strip():
                    logger.info("Successfully downloaded API key from HuggingFace")
                    # Persist it immediately
                    self._persist_api_key(key)
            except Exception as e:
                logger.warning(f"Failed to download API key from HuggingFace: {e}")

        # In local mode, wait for user to configure settings and click "Start"
        if is_local and self._settings_app is not None:
            logger.info("Local mode: waiting for user to configure settings via UI and click Start...")
            self._start_event.wait()  # blocks until POST /start_app is called
            logger.info("Start signal received, launching pipeline...")

        self._pipeline_started = True

        # Start media after key is set/available
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start

        async def runner() -> None:
            # Capture loop for cross-thread personality actions
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop  # type: ignore[assignment]
            # Mount personality routes now that loop and handler are available
            try:
                if self._settings_app is not None:
                    mount_personality_routes(
                        self._settings_app,
                        self.handler,
                        lambda: self._asyncio_loop,
                        persist_personality=self._persist_personality,
                        get_persisted_personality=self._read_persisted_personality,
                    )
            except Exception:
                pass
            self._tasks = [
                asyncio.create_task(self.handler.start_up(), name="openai-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await self.handler.shutdown()

        asyncio.run(runner())

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Stops audio recording and playback first
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)
        """
        logger.info("Stopping LocalStream...")

        # Stop media pipelines FIRST before cancelling async tasks
        # This ensures clean shutdown before PortAudio cleanup
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug(f"Error stopping recording (may already be stopped): {e}")

        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug(f"Error stopping playback (may already be stopped): {e}")

        # Now signal async loops to stop
        self._stop_event.set()

        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def clear_audio_queue(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        logger.info("User intervention: flushing player queue")
        if self._robot.media.backend == MediaBackend.GSTREAMER:
            # Directly flush gstreamer audio pipe
            self._robot.media.audio.clear_player()
        elif self._robot.media.backend == MediaBackend.DEFAULT or self._robot.media.backend == MediaBackend.DEFAULT_NO_VIDEO:
            self._robot.media.audio.clear_output_buffer()
        self.handler.output_queue = asyncio.Queue()

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.debug(f"Audio recording started at {input_sample_rate} Hz")

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)  # avoid busy loop

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        while not self._stop_event.is_set():
            handler_output = await self.handler.emit()

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content if len(content) < 500 else content[:500] + "…",
                        )

            elif isinstance(handler_output, tuple):
                input_sample_rate, audio_data = handler_output
                output_sample_rate = self._robot.media.get_output_audio_samplerate()

                # Reshape if needed
                if audio_data.ndim == 2:
                    if audio_data.size == 0:
                        continue
                    # Scipy channels last convention
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    # Multiple channels -> Mono channel
                    if audio_data.shape[1] > 1:
                        audio_data = audio_data[:, 0]

                # Cast if needed
                audio_frame = audio_to_float32(audio_data)

                # Resample if needed
                if input_sample_rate != output_sample_rate:
                    audio_frame = resample(
                        audio_frame,
                        int(len(audio_frame) * output_sample_rate / input_sample_rate),
                    )

                self._robot.media.push_audio_sample(audio_frame)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
