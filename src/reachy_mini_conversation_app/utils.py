import logging
import argparse
import warnings
from typing import Any, Tuple, Optional

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.camera_worker import CameraWorker


def parse_args() -> Tuple[argparse.Namespace, list]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--smolvlm",
        default=False,
        action="store_true",
        help="Use SmolVLM local vision model for periodic scene description.",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--dashboard", default=False, action="store_true", help="Launch the settings dashboard on port 8000"
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="[Optional] Robot name/prefix for Zenoh topics (must match daemon's --robot-name). Only needed for development with multiple robots.",
    )
    parser.add_argument(
        "--openai-realtime",
        default=False,
        action="store_true",
        help="Use OpenAI Realtime API instead of local processing.",
    )
    return parser.parse_known_args()


def handle_vision_stuff(args: argparse.Namespace, current_robot: ReachyMini) -> Tuple[CameraWorker | None, Any]:
    """Initialize camera worker and vision manager.

    By default, vision is handled by the LLM via camera tool.
    If --smolvlm flag is used, SmolVLM will process images periodically.
    """
    camera_worker = None
    vision_manager = None

    if not args.no_camera:
        # Initialize camera worker
        camera_worker = CameraWorker(current_robot)

        # Initialize Vision Manager (On-Demand Mode by default)
        # This allows tools like 'camera' to use the configured Local VLM without
        # running a continuous background process.
        try:
            from reachy_mini_conversation_app.vision.processors import initialize_vision_manager

            # continuous_mode=False ensures we don't run the heavy background loop
            vision_manager = initialize_vision_manager(camera_worker, continuous_mode=False)
            if vision_manager:
                logging.getLogger(__name__).info("Vision Manager initialized (On-Demand Mode).")
        except ImportError:
            logging.getLogger(__name__).warning(
                "Vision dependencies missing, camera tool may not work for description."
            )
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to init vision: {e}")

    return camera_worker, vision_manager


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("fastrtc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("fastrtc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger


def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")

    if robot_name is not None:
        logger.error(f"  2. Daemon must be started with: --robot-name '{robot_name}'")
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
