"""Test qwen3-vl:4b on Ollama with and without thinking mode.

Compares three approaches:
  1. Default (thinking ON) — baseline
  2. /no_think in system prompt — may not work on all Ollama versions
  3. /no_think in user message — the correct approach per Qwen3 chat template

Usage:
    uv run python scripts/test_qwen3_no_think.py
    uv run python scripts/test_qwen3_no_think.py --url http://jetson:8000/v1
"""

import time
import asyncio
import argparse

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam


TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": "Make the robot speak out loud.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to speak."}},
                "required": ["text"],
            },
        },
    },
]


async def run_test(client: AsyncOpenAI, model: str, label: str, system: str, user: str):
    """Run a single streaming chat completion and print timing + output."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print(f"System: {system[:80]}{'...' if len(system) > 80 else ''}")
    print(f"User: {user[:80]}{'...' if len(user) > 80 else ''}")
    print("-" * 60)

    t_start = time.monotonic()
    t_first_token = None
    full_text = ""
    tool_calls: dict[int, dict[str, str]] = {}

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=TOOLS,
            stream=True,
            parallel_tool_calls=False,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                if t_first_token is None:
                    t_first_token = time.monotonic()
                full_text += delta.content

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"name": "", "arguments": ""}
                    if tc.function and tc.function.name:
                        tool_calls[idx]["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments

        t_end = time.monotonic()
        total_ms = (t_end - t_start) * 1000
        ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else None

        print(f"\nTTFT: {ttft_ms:.0f}ms" if ttft_ms else "\nTTFT: N/A (no text)")
        print(f"Total: {total_ms:.0f}ms")

        if full_text:
            if "<think>" in full_text:
                think_end = full_text.find("</think>")
                if think_end > 0:
                    thinking = full_text[full_text.find("<think>") + 7 : think_end].strip()
                    actual = full_text[think_end + 8 :].strip()
                    print(f"THINKING ({len(thinking)} chars): {thinking[:150]}...")
                    print(f"RESPONSE: {actual[:200]}")
                    print(f">>> THINKING MODE IS ON — {len(thinking)} chars wasted")
                else:
                    print(f"TEXT: {full_text[:300]}")
                    print(">>> Open <think> tag — model is thinking")
            else:
                print(f"RESPONSE: {full_text[:300]}")
                print(">>> No thinking tags — GOOD")

        if tool_calls:
            for idx in sorted(tool_calls):
                call = tool_calls[idx]
                print(f"TOOL[{idx}]: {call['name']}({call['arguments'][:100]})")

        if not full_text and not tool_calls:
            print("(empty response)")

        return ttft_ms

    except Exception as e:
        print(f"ERROR: {e}")
        return None


async def main():
    """Run thinking-mode tests against different prompt strategies."""
    parser = argparse.ArgumentParser(description="Test qwen3-vl thinking mode")
    parser.add_argument("--url", default="http://localhost:11434/v1", help="OpenAI-compatible API URL")
    parser.add_argument("--model", default="qwen3-vl:4b", help="Model name")
    parser.add_argument("--api-key", default="ollama", help="API key")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.url, api_key=args.api_key)
    model = args.model

    print(f"Server: {args.url}")
    print(f"Model: {model}")
    print("\nWarm-up call first (ignore timing)...")

    # Warm up
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
        print(
            f"Warm-up done: {resp.choices[0].message.content[:50] if resp.choices[0].message.content else '(empty)'}..."
        )
    except Exception as e:
        print(f"Warm-up failed: {e}")
        return

    results = {}

    # Test 1: Default (thinking ON)
    results["default"] = await run_test(
        client,
        model,
        "DEFAULT — thinking likely ON",
        "You are a friendly robot. Keep responses to one sentence.",
        "Hello, how are you?",
    )

    # Test 2: /no_think in system prompt
    results["system"] = await run_test(
        client,
        model,
        "/no_think in SYSTEM PROMPT",
        "/no_think\nYou are a friendly robot. Keep responses to one sentence.",
        "Hello, how are you?",
    )

    # Test 3: /no_think in user message
    results["user"] = await run_test(
        client,
        model,
        "/no_think in USER MESSAGE — should work per Qwen3 template",
        "You are a friendly robot. Keep responses to one sentence.",
        "/no_think\nHello, how are you?",
    )

    # Test 4: /no_think in user message + tool calling
    results["user+tools"] = await run_test(
        client,
        model,
        "/no_think in USER MESSAGE + TOOL CALL",
        "You are a friendly robot. Always use the speak tool to talk.",
        "/no_think\nSay hello to the baby",
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY (TTFT comparison)")
    print("-" * 60)
    for label, ttft in results.items():
        status = f"{ttft:.0f}ms" if ttft else "N/A"
        print(f"  {label:20s}: {status}")
    print("\nIf /no_think works, the user-message tests should have much lower TTFT.")


if __name__ == "__main__":
    asyncio.run(main())
