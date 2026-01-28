import logging
import json
from typing import AsyncGenerator, Dict, Any, List
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class LocalLLM:
    """Wrapper for Local LLM (Ollama) via OpenAI compatible API."""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "ministral-3b", system_prompt: str = ""):
        self.client = AsyncOpenAI(base_url=base_url, api_key="ollama")
        self.model = model
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        # Reset history or just update the system message if it exists?
        # For simplicity, we keep history but ensure system prompt is used in next call
        
    async def chat_stream(self, user_text: str) -> AsyncGenerator[str, None]:
        """Send message and yield response chunks."""
        
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )

            full_response = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content

            # Update history
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            logger.error(f"LLM Chat Error: {e}")
            yield f"[Error: {e}]"
