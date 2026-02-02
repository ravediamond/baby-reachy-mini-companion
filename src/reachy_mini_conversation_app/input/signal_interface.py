import subprocess
import json
import logging
import asyncio
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

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
                    self.username = lines[0]
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

    async def send_message(self, text: str, recipient: str) -> bool:
        """Send a message via signal-cli."""
        if not self.available or not self.username:
            logger.warning(f"[MOCK] Signal send to {recipient}: {text}")
            return True

        cmd = [
            "signal-cli", "-u", self.username, "send", "-m", text, recipient
        ]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                logger.info(f"Signal sent to {recipient}")
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

        # receive --json returns NDJSON
        cmd = ["signal-cli", "-u", self.username, "receive", "--json"]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                # Often returns non-zero if no messages? Or if error.
                # logger.debug(f"Signal poll stderr: {stderr.decode()}")
                return []
            
            messages = []
            output = stdout.decode()
            if not output.strip():
                return []

            for line in output.splitlines():
                try:
                    data = json.loads(line)
                    envelope = data.get("envelope", {})
                    source = envelope.get("source") or envelope.get("sourceNumber")
                    
                    # Check for data message (text)
                    if "dataMessage" in envelope and envelope["dataMessage"]:
                        msg_text = envelope["dataMessage"].get("message")
                        if msg_text and source:
                            messages.append({
                                "sender": source,
                                "content": msg_text,
                                "type": "text"
                            })
                except json.JSONDecodeError:
                    continue
            
            return messages

        except Exception as e:
            logger.error(f"Signal poll error: {e}")
            return []
