import argparse
import asyncio
from threading import Thread

from reachy_mini.utils.camera import find_camera

from reachy_mini_conversation_demo.camera_worker import CameraWorker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation Demo")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    parser.add_argument(
        "--head-tracker",
        choices=["yolo", "mediapipe", None],
        default=None,
        help="Choose head tracker (default: mediapipe)",
    )
    parser.add_argument(
        "--no-camera", default=False, action="store_true", help="Disable camera usage"
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Run in headless mode"
    )
    return parser.parse_args()


def handle_vision_stuff(args, current_robot):
    """Initialize camera, head tracker and camera worker."""
    camera = None
    camera_worker = None
    head_tracker = None
    vision_manager = None
    if not args.no_camera:
        if not args.sim:
            camera = find_camera()
        else:
            import cv2

            camera = cv2.VideoCapture(0)

        print("camera", camera)
        if args.head_tracker is not None:
            if args.head_tracker == "yolo":
                from reachy_mini_conversation_demo.vision.yolo_head_tracker import (
                    HeadTracker,
                )

                head_tracker = HeadTracker()

            elif args.head_tracker == "mediapipe":
                from reachy_mini_toolbox.vision import HeadTracker

                head_tracker = HeadTracker()

        print("head tracker", head_tracker)

        camera_worker = CameraWorker(camera, current_robot, head_tracker)

    return camera, camera_worker, head_tracker, vision_manager


class AioTaskThread:
    """Lance UNE coroutine dans son propre thread + event loop."""

    def __init__(self, coro_fn, *args, **kwargs):
        self.coro_fn = coro_fn
        self.args = args
        self.kwargs = kwargs
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self._run, daemon=True)
        self._stop_async: asyncio.Event | None = None

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self._stop_async = asyncio.Event()

        async def runner():
            await self.coro_fn(*self.args, self._stop_async, **self.kwargs)

        try:
            self.loop.run_until_complete(runner())
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    def start(self):
        self.thread.start()

    def request_stop(self):
        if self._stop_async is not None:
            self.loop.call_soon_threadsafe(self._stop_async.set)

    def join(self):
        self.thread.join()
