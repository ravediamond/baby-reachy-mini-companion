import subprocess
import json
import logging
import asyncio
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def _extract_phone_number(text: str) -> Optional[str]:
    """Extract phone number from signal-cli output."""
    # Match phone numbers like +33650924838
    match = re.search(r'(\+\d+)', text)
    return match.group(1) if match else None


class SignalInterface:
    """Interface for signal-cli."""

    def __init__(self, username: Optional[str] = None):
        """Initialize the interface.

        Args:
            username: The registered phone number (with country code).
                      If None, tries to find one or logs a warning.
        """
        self.username = username
        self.available = self._check_availability()

        if self.available and not self.username:
            # Try to auto-detect
            try:
                res = subprocess.run(["signal-cli", "listAccounts"], capture_output=True, text=True)
                lines = res.stdout.strip().splitlines()
                if lines:
                    # Extract just the phone number from "Number: +33..." format
                    self.username = _extract_phone_number(lines[0])
                    if self.username:
                        logger.info(f"Auto-detected Signal account: {self.username}")
            except Exception:
                pass

    def _check_availability(self) -> bool:
        try:
            subprocess.run(["signal-cli", "--version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("signal-cli not found. Signal features will be disabled/mocked.")
            return False

    async def send_message(self, text: str, recipient: str, attachment: Optional[str] = None) -> bool:
        """Send a message via signal-cli.

        Args:
            text: The message text.
            recipient: The recipient phone number.
            attachment: Optional path to a file to attach.
        """
        if not self.available or not self.username:
            logger.warning(f"[MOCK] Signal send to {recipient}: {text}")
            return True

        # Build command: signal-cli -u USER send -m TEXT [-a FILE] [--note-to-self | RECIPIENT]
        cmd = ["signal-cli", "-u", self.username, "send", "-m", text]

        if attachment:
            cmd.extend(["-a", attachment])

        # If sending to self, use --note-to-self flag
        if recipient == self.username:
            cmd.append("--note-to-self")
        else:
            cmd.append(recipient)

        logger.debug(f"Signal cmd: {' '.join(cmd)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                logger.info(f"Signal sent to {recipient}" + (" with attachment" if attachment else ""))
                return True
            else:
                logger.error(f"Signal send failed: {stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Signal send error: {e}")
            return False

    async def poll_messages(self) -> List[Dict[str, str]]:
        """Poll for new messages."""
        if not self.available or not self.username:
            return []

        # signal-cli 0.13.x outputs plain text, not JSON
        cmd = ["signal-cli", "-u", self.username, "receive"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            output = stdout.decode()
            if not output.strip():
                return []

            messages = []
            current_sender = None
            current_body = None

            for line in output.splitlines():
                # Parse envelope header: Envelope from: "Name" +1234567890 (device: 1)
                if line.startswith("Envelope from:"):
                    # Extract phone number
                    current_sender = _extract_phone_number(line)
                    current_body = None

                # Parse message body
                elif line.strip().startswith("Body:"):
                    current_body = line.strip()[5:].strip()  # Remove "Body:" prefix

                    if current_sender and current_body:
                        # Skip messages from ourselves (sync messages we sent)
                        # unless it's a "Note to Self" which we want to process
                        messages.append({
                            "sender": current_sender,
                            "content": current_body,
                            "type": "text"
                        })
                        current_body = None

            return messages

        except Exception as e:
            logger.error(f"Signal poll error: {e}")
            return []
