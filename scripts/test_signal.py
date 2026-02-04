#!/usr/bin/env python3
"""Test Signal polling and messaging."""

import asyncio
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reachy_mini_conversation_app.input.signal_interface import SignalInterface
from reachy_mini_conversation_app.config import config


async def main():
    print("Signal Test")
    print("-" * 40)

    signal = SignalInterface()
    print(f"Available: {signal.available}")
    print(f"Account: {signal.username}")
    print(f"Target: {config.SIGNAL_USER_PHONE}")
    print("-" * 40)

    if not signal.available:
        print("Signal CLI not available!")
        return

    target = config.SIGNAL_USER_PHONE
    if not target:
        print("No SIGNAL_USER_PHONE configured!")
        return

    # Test 1: Send text
    print("\n[1] Sending text message...")
    success = await signal.send_message("Hello from Reachy Mini!", target)
    print(f"    Result: {success}")

    # Test 2: Send image
    print("\n[2] Sending image...")
    try:
        import cv2
        import numpy as np

        # Create test image
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        cv2.putText(img, "Reachy Mini", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            temp_path = f.name

        success = await signal.send_message("Test photo", target, attachment=temp_path)
        print(f"    Result: {success}")

        os.unlink(temp_path)
    except Exception as e:
        print(f"    Error: {e}")

    # Test 3: Poll messages
    print("\n[3] Polling messages (send a message to test)...")
    for i in range(3):
        messages = await signal.poll_messages()
        if messages:
            for m in messages:
                print(f"    >> {m['sender']}: {m['content']}")
        else:
            print(f"    (poll {i+1}: no messages)")
        await asyncio.sleep(2)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
