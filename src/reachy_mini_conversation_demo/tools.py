from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import cv2
import numpy as np

from reachy_mini_conversation_demo.vision import VisionManager

logger = logging.getLogger(__name__)

# Types & state

Direction = Literal["left", "right", "up", "down", "front"]


@dataclass
class Deps:
    """External dependencies the tools need"""

    reachy_mini: Any
    create_head_pose: Any
    camera: cv2.VideoCapture
    # Optional deps
    vision_manager: Optional[VisionManager] = None


# Helpers
def _encode_jpeg_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _read_frame(cap: cv2.VideoCapture, attempts: int = 5) -> np.ndarray:
    """Grab a frame with a small retry."""
    trials, frame, ret = 0, None, False
    while trials < attempts and not ret:
        ret, frame = cap.read()
        trials += 1
        if not ret and trials < attempts:
            time.sleep(0.1)  # Small delay between retries
    if not ret or frame is None:
        logger.error("Failed to capture image from camera after %d attempts", attempts)
        raise RuntimeError("Failed to capture image from camera.")
    return frame


# Tool coroutines
async def move_head(deps: Deps, *, direction: Direction) -> Dict[str, Any]:
    """Move your head in a given direction"""
    logger.info("Tool call: move_head direction=%s", direction)

    # Import and update the SAME global variables that main.py reads
    from reachy_mini_conversation_demo.main import movement_manager

    if direction == "left":
        target = deps.create_head_pose(0, 0, 0, 0, 0, 40, degrees=True)
    elif direction == "right":
        target = deps.create_head_pose(0, 0, 0, 0, 0, -40, degrees=True)
    elif direction == "up":
        target = deps.create_head_pose(0, 0, 0, 0, -30, 0, degrees=True)
    elif direction == "down":
        target = deps.create_head_pose(0, 0, 0, 0, 30, 0, degrees=True)
    else:  # front
        target = deps.create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

    movement_manager.moving_start = time.monotonic()
    movement_manager.moving_for = 1.0
    movement_manager.current_head_pose = target

    # Start the movement
    deps.reachy_mini.goto_target(target, duration=1.0)

    return {"status": f"looking {direction}"}


async def head_tracking(deps: Deps, *, start: bool) -> Dict[str, Any]:
    """Toggle head tracking state"""
    from reachy_mini_conversation_demo.main import movement_manager

    movement_manager.is_head_tracking_on = bool(start)
    status = "started" if start else "stopped"
    logger.info("Tool call: head_tracking %s", status)
    return {"status": f"head tracking {status}"}


async def camera(deps: Deps, *, question: str) -> Dict[str, Any]:
    """
    Capture an image and ask a question about it using local SmolVLM2.
    Returns: {"image_description": '...'} or {"error": '...'}.
    """
    q = (question or "").strip()
    if not q:
        logger.error("camera: empty question")
        return {"error": "question must be a non-empty string"}

    logger.info("Tool call: camera question=%s", q[:120])

    try:
        frame = await asyncio.to_thread(_read_frame, deps.camera)
    except Exception as e:
        logger.exception("camera: failed to capture image")
        return {"error": f"camera capture failed: {type(e).__name__}: {e}"}

    if not deps.vision_manager:
        logger.error("camera: vision manager not available")
        return {"error": "vision processing not available"}

    # Optional sound effect
    # try:
    #     # TODO Mute mic while hmmm
    #     deps.reachy_mini.play_sound(f"hmm{np.random.randint(1, 6)}.wav")
    # except Exception:
    #     logger.debug("camera: optional play_sound failed", exc_info=True)

    try:
        desc = await asyncio.to_thread(
            deps.vision_manager.processor.process_image, frame, q
        )
        logger.debug(
            "camera: SmolVLM2 result length=%d",
            len(desc) if isinstance(desc, str) else -1,
        )
        return {"image_description": desc}
    except Exception as e:
        logger.exception("camera: vision pipeline error")
        return {"error": f"vision failed: {type(e).__name__}: {e}"}


async def describe_current_scene(deps: Deps) -> Dict[str, Any]:
    """Get current scene description from camera with detailed analysis"""
    logger.info("Tool call: describe_current_scene")

    if not deps.vision_manager:
        return {"error": "Vision processing not available"}

    # Ensure processor is initialized
    if not deps.vision_manager.processor._initialized:
        if not deps.vision_manager.processor.initialize():
            return {"error": "Failed to initialize vision processor"}

    try:
        result = await deps.vision_manager.process_current_frame(
            "Describe what you currently see in detail, focusing on people, objects, and activities."
        )
        return result
    except Exception as e:
        logger.exception("Failed to describe current scene")
        return {"error": f"Scene description failed: {type(e).__name__}: {e}"}


async def get_scene_context(deps: Deps) -> Dict[str, Any]:
    """Get the most recent automatic scene description for context"""
    logger.info("Tool call: get_scene_context")

    if not deps.vision_manager:
        return {"error": "Vision processing not available"}

    try:
        description = await deps.vision_manager.get_current_description()
        if not description:
            return {
                "context": "No scene description available yet",
                "note": "Vision processing may still be initializing",
            }

        return {
            "context": description,
            "note": "This is from periodic automatic scene analysis",
        }
    except Exception as e:
        logger.exception("Failed to get scene context")
        return {"error": f"Scene context failed: {type(e).__name__}: {e}"}


async def analyze_scene_for(deps: Deps, *, purpose: str = "general") -> Dict[str, Any]:
    """Analyze current scene for specific purpose"""
    logger.info("Tool call: analyze_scene_for purpose=%s", purpose)

    if not deps.vision_manager:
        return {"error": "Vision processing not available"}

    try:
        # Custom prompts based on purpose
        prompts = {
            "safety": "Look for any safety concerns, obstacles, or hazards in the scene.",
            "people": "Describe any people you see, their positions and what they're doing.",
            "objects": "Identify and describe the main objects and items visible in the scene.",
            "activity": "Describe what activities or actions are happening in the scene.",
            "navigation": "Describe the space for navigation - obstacles, pathways, and layout.",
            "general": "Provide a general description of the scene including people, objects, and activities.",
        }

        prompt = prompts.get(purpose.lower(), prompts["general"])

        result = await deps.vision_manager.process_current_frame(prompt)
        result["analysis_purpose"] = purpose

        return result
    except Exception as e:
        logger.exception("Failed to analyze scene for %s", purpose)
        return {"error": f"Scene analysis failed: {type(e).__name__}: {e}"}


# Registration helpers
TOOL_SPECS = [
    {
        "type": "function",
        "name": "move_head",
        "description": "Move your head in a given direction: left, right, up, down or front.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down", "front"],
                }
            },
            "required": ["direction"],
        },
    },
    {
        "type": "function",
        "name": "camera",
        "description": "Take a picture using your camera, ask a question about the picture. Get an answer about the picture",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask about the picture",
                }
            },
            "required": ["question"],
        },
    },
    # {
    #     "type": "function",
    #     "name": "head_tracking",
    #     "description": "Start or stop head tracking",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "start": {
    #                 "type": "boolean",
    #                 "description": "Whether to start or stop head tracking",
    #             }
    #         },
    #         "required": ["start"],
    #     },
    # },
    # {
    #     "type": "function",
    #     "name": "describe_current_scene",
    #     "description": "Get a detailed description of what you currently see through your camera",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {},
    #         "required": []
    #     }
    # },
    {
        "type": "function",
        "name": "get_scene_context",
        "description": "Get the most recent automatic scene description for conversational context",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    # {
    #     "type": "function",
    #     "name": "analyze_scene_for",
    #     "description": "Analyze the current scene for a specific purpose (safety, people, objects, activity, navigation, or general)",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "purpose": {
    #                 "type": "string",
    #                 "enum": ["safety", "people", "objects", "activity", "navigation", "general"],
    #                 "description": "The specific purpose for scene analysis"
    #             }
    #         },
    #         "required": ["purpose"]
    #     }
    # }
]


def get_tool_registry(deps: Deps):
    """Map tool name -> coroutine that accepts **kwargs (tool args)."""
    return {
        "move_head": lambda **kw: move_head(deps, **kw),
        "camera": lambda **kw: camera(deps, **kw),
        "head_tracking": lambda **kw: head_tracking(deps, **kw),
        "describe_current_scene": lambda **kw: describe_current_scene(deps),
        "get_scene_context": lambda **kw: get_scene_context(deps),
        "analyze_scene_for": lambda **kw: analyze_scene_for(deps, **kw),
    }


async def dispatch_tool_call(name: str, args_json: str, deps: Deps) -> Dict[str, Any]:
    """Utility to execute a tool from streamed function_call arguments."""
    try:
        args = json.loads(args_json or "{}")
    except Exception:
        args = {}
    registry = get_tool_registry(deps)
    func = registry.get(name)
    if not func:
        return {"error": f"unknown tool: {name}"}
    try:
        return await func(**args)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.exception("Tool error in %s: %s", name, error_msg)
        return {"error": error_msg}
