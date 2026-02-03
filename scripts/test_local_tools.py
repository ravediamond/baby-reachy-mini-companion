import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from reachy_mini_conversation_app.local.llm import LocalLLM
from reachy_mini_conversation_app.tools.core_tools import get_tool_specs
from reachy_mini_conversation_app.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ToolTest")


async def test_tools():
    # 1. Initialize LLM
    llm_url = "http://localhost:11434/v1"
    llm_model = config.LOCAL_LLM_MODEL or "ministral-3:3b"

    logger.info(f"Connecting to Ollama at {llm_url} using model {llm_model}...")

    llm = LocalLLM(base_url=llm_url, model=llm_model)

    # 2. Get Tools
    logger.info("Loading tools for profile...")
    tools = get_tool_specs()
    logger.info(f"Loaded {len(tools)} tools: {[t['function']['name'] for t in tools]}")

    # 3. Test Cases
    test_prompts = [
        ("Look at the camera.", "camera"),
        ("Stop dancing immediately.", "stop_dance"),
        ("Do nothing for a bit.", "do_nothing"),
        ("Show me a happy face.", "play_emotion"),
    ]

    llm.set_system_prompt(
        "You are a robot assistant. Use the available tools to control your body when requested by the user. If the user asks for a specific action, call the corresponding tool."
    )

    for prompt, expected_tool in test_prompts:
        print(f"\n\n--- Testing Prompt: '{prompt}' (Expect: {expected_tool}) ---")

        tool_detected = False
        response_text = ""

        async for event in llm.chat_stream(user_text=prompt, tools=tools):
            if event["type"] == "tool_call":
                tc = event["tool_call"]
                print(f"✅ Tool Call Detected: {tc['function']['name']}")
                print(f"   Arguments: {tc['function']['arguments']}")
                tool_detected = True
            elif event["type"] == "text":
                response_text += event["content"]
            elif event["type"] == "error":
                print(f"❌ Error: {event['content']}")

        if not tool_detected:
            print(f"⚠️ No tool call detected. Response: {response_text.strip()}")


if __name__ == "__main__":
    asyncio.run(test_tools())
