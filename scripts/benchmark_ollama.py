#!/usr/bin/env python3
# mypy: ignore-errors
"""Benchmark Ollama response times with and without tools."""

import sys
import time
import asyncio
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai import AsyncOpenAI

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import get_tool_specs


async def benchmark():
    """Run LLM benchmark against Ollama."""
    model = config.LOCAL_LLM_MODEL or "ministral-3:3b"
    base_url = "http://localhost:11434/v1"

    print("Benchmarking Ollama")
    print(f"  Model: {model}")
    print(f"  URL: {base_url}")
    print("-" * 50)

    client = AsyncOpenAI(base_url=base_url, api_key="ollama")

    # Load tools
    tools = get_tool_specs()
    print(f"  Tools loaded: {len(tools)}")
    for t in tools:
        print(f"    - {t['function']['name']}")
    print("-" * 50)

    # PRE-WARM: Send a dummy request with tools to warm up
    print("\n[PRE-WARM] Warming up LLM with tools...")
    start = time.time()
    await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hi"}], tools=tools, stream=False
    )
    print(f"  Warm-up completed in {time.time() - start:.2f}s")
    print("-" * 50)

    test_prompts = [
        "Hello",
        "What are you?",
        "Tell me a short joke",
    ]

    # Test WITHOUT tools
    print("\n[TEST 1] Without tools:")
    for prompt in test_prompts:
        start = time.time()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep responses brief."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        elapsed = time.time() - start
        content = (
            response.choices[0].message.content[:50] + "..."
            if len(response.choices[0].message.content) > 50
            else response.choices[0].message.content
        )
        print(f"  '{prompt}' -> {elapsed:.2f}s | {content}")

    # Test WITH tools
    print("\n[TEST 2] With tools:")
    for prompt in test_prompts:
        start = time.time()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep responses brief."},
                {"role": "user", "content": prompt},
            ],
            tools=tools,
            stream=False,
        )
        elapsed = time.time() - start
        msg = response.choices[0].message
        if msg.tool_calls:
            content = f"[TOOL: {msg.tool_calls[0].function.name}]"
        else:
            content = msg.content[:50] + "..." if msg.content and len(msg.content) > 50 else msg.content
        print(f"  '{prompt}' -> {elapsed:.2f}s | {content}")

    # Test WITH tools - streaming (time to first token)
    print("\n[TEST 3] With tools (streaming - time to first token):")
    for prompt in test_prompts:
        start = time.time()
        first_token_time = None
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep responses brief."},
                {"role": "user", "content": prompt},
            ],
            tools=tools,
            stream=True,
        )
        tokens = []
        async for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time() - start
            delta = chunk.choices[0].delta
            if delta.content:
                tokens.append(delta.content)

        total_time = time.time() - start
        content = "".join(tokens)[:50] + "..." if len("".join(tokens)) > 50 else "".join(tokens)
        print(f"  '{prompt}' -> first: {first_token_time:.2f}s, total: {total_time:.2f}s | {content}")

    # Test tool-calling prompts
    tool_prompts = [
        "Send me a message saying hello",
        "Send me a photo",
        "Look at the camera",
        "Do nothing for a bit",
    ]

    print("\n[TEST 4] Tool-calling prompts (should trigger tool calls):")
    for prompt in tool_prompts:
        start = time.time()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a robot. When user asks to send a message, call send_signal tool. When user asks for a photo, call send_signal_photo tool. Call tools directly.",
                },
                {"role": "user", "content": prompt},
            ],
            tools=tools,
            stream=False,
        )
        elapsed = time.time() - start
        msg = response.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            content = f"[TOOL: {tc.function.name}({tc.function.arguments})]"
        else:
            content = (
                f"[TEXT: {msg.content[:40]}...]" if msg.content and len(msg.content) > 40 else f"[TEXT: {msg.content}]"
            )
        print(f"  '{prompt}' -> {elapsed:.2f}s | {content}")

    print("\n[TEST 5] Tool-calling prompts (streaming):")
    for prompt in tool_prompts:
        start = time.time()
        first_token_time = None
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a robot. When user asks to send a message, call send_signal tool. When user asks for a photo, call send_signal_photo tool. Call tools directly.",
                },
                {"role": "user", "content": prompt},
            ],
            tools=tools,
            stream=True,
        )

        tokens = []
        tool_calls_buffer = {}

        async for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time() - start
            delta = chunk.choices[0].delta
            if delta.content:
                tokens.append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {"name": "", "args": ""}
                    if tc.function:
                        if tc.function.name:
                            tool_calls_buffer[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_buffer[idx]["args"] += tc.function.arguments

        total_time = time.time() - start

        if tool_calls_buffer:
            tc = list(tool_calls_buffer.values())[0]
            content = f"[TOOL: {tc['name']}({tc['args']})]"
        else:
            text = "".join(tokens)
            content = f"[TEXT: {text[:40]}...]" if len(text) > 40 else f"[TEXT: {text}]"

        print(f"  '{prompt}' -> first: {first_token_time:.2f}s, total: {total_time:.2f}s | {content}")

    print("\n" + "-" * 50)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(benchmark())
