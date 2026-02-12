"""Entrypoint for the Reachy Mini conversation app."""

import sys
import time
import asyncio
import argparse
import threading
import webbrowser
from typing import Any, Dict, Optional
from pathlib import Path

from fastapi import FastAPI

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
    log_connection_troubleshooting,
)


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: Optional[ReachyMini] = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.console import LocalStream
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(f"Connection timeout: Failed to connect to Reachy Mini daemon. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(f"Connection failed: Unable to establish connection to Reachy Mini. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(f"Unexpected error during robot initialization: {type(e).__name__}: {e}")
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    camera_worker, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    audio_classifier_status: Dict[str, Any] = {"latest_event": None, "timestamp": 0.0}
    vision_threat_status: Dict[str, Any] = {"latest_threat": None, "timestamp": 0.0, "objects": []}

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
        audio_classifier_status=audio_classifier_status,
        vision_threat_status=vision_threat_status,
    )

    # Launch a standalone settings dashboard when --dashboard is passed
    if args.dashboard and settings_app is None:
        settings_app = FastAPI(title="Reachy Mini Settings")
        if instance_path is None:
            instance_path = str(Path.cwd())

    from reachy_mini_conversation_app.local.handler import LocalSessionHandler

    logger.info("Using Local LLM (fully local + Signal)")
    handler: Any = LocalSessionHandler(deps)

    # Headless mode: wire settings_app + instance_path to console LocalStream
    # Routes are registered in __init__ so the dashboard is ready immediately.
    stream_manager = LocalStream(
        handler,
        robot,
        settings_app=settings_app,
        instance_path=instance_path,
    )

    # Start uvicorn AFTER routes are registered
    if args.dashboard and settings_app is not None:
        import uvicorn

        def _run_settings_server() -> None:
            uvicorn.run(settings_app, host="0.0.0.0", port=8321, log_level="warning")

        threading.Thread(target=_run_settings_server, daemon=True).start()
        logger.info("Settings dashboard available at http://localhost:8321")
        try:
            threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8321")).start()
        except Exception:
            pass  # headless environments (e.g. Jetson) may not have a browser

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        try:
            threading.Timer(1.5, lambda: webbrowser.open("http://localhost:7860")).start()
        except Exception:
            pass  # headless environments may not have a browser

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
