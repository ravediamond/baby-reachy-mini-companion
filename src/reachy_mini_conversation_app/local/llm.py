import logging
import json
from typing import AsyncGenerator, Dict, Any, List, Optional
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

logger = logging.getLogger(__name__)

class LocalLLM:
    """Wrapper for Local LLM (Ollama) via OpenAI compatible API."""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "qwen2.5:3b", system_prompt: str = ""):
        self.client = AsyncOpenAI(base_url=base_url, api_key="ollama")
        self.model = model
        self.system_prompt = system_prompt
        self.history: List[Dict[str, Any]] = []

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        
    async def chat_stream(
        self, 
        user_text: Optional[str] = None, 
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_outputs: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send message and yield response events (text or tool calls)."""
        
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        
        if user_text:
            messages.append({"role": "user", "content": user_text})
            # Temporarily add to history (will be confirmed after success)
            # Actually, better to manage history at the end of the turn
        
        if tool_outputs:
            # Append tool outputs to messages (and history ideally)
            # Assuming tool_outputs are valid tool messages
            messages.extend(tool_outputs)

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=True
            )

            full_content = ""
            tool_calls_buffer: Dict[int, Any] = {}

            async for chunk in stream:
                delta = chunk.choices[0].delta
                
                # 1. Handle Text Content
                if delta.content:
                    full_content += delta.content
                    yield {"type": "text", "content": delta.content}

                # 2. Handle Tool Calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        
                        if tc.id:
                            tool_calls_buffer[idx]["id"] = tc.id
                        
                        if tc.function:
                            if tc.function.name:
                                tool_calls_buffer[idx]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_buffer[idx]["function"]["arguments"] += tc.function.arguments

            # End of stream processing
            
            # Commit user message to history if it was new
            if user_text:
                self.history.append({"role": "user", "content": user_text})
            
            # Commit tool outputs if any
            if tool_outputs:
                self.history.extend(tool_outputs)

            # Check if we have tool calls to yield
            final_tool_calls = []
            if tool_calls_buffer:
                for idx in sorted(tool_calls_buffer.keys()):
                    tool_call = tool_calls_buffer[idx]
                    final_tool_calls.append(tool_call)
                    yield {"type": "tool_call", "tool_call": tool_call}
                
                # Append assistant message with tool calls to history
                self.history.append({
                    "role": "assistant",
                    "content": full_content if full_content else None,
                    "tool_calls": final_tool_calls
                })
            else:
                # Normal text response
                self.history.append({"role": "assistant", "content": full_content})

        except Exception as e:
            logger.error(f"LLM Chat Error: {e}")
            yield {"type": "error", "content": str(e)}
