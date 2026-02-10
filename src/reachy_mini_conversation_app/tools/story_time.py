import random
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


# Library of stories with emotional/prosodic markers for Kokoro TTS
STORIES = {
    "three_little_pigs": (
        "[STORY] Once upon a time, there were three little pigs."
        " [HAPPY] They set out to build their own houses!"
        " The first pig built his house of straw. [sad] It was not very strong."
        " The second pig built his house of sticks. [sad] It was a little better, but still shaky."
        " The third pig built his house of bricks. [HAPPY] It was strong and safe!"
        " Then, the Big Bad Wolf came! [sad] He huffed, and he puffed, and he blew the straw house down!"
        " He went to the stick house. He huffed, and he puffed, and he blew that house down too!"
        " [HAPPY] But when he came to the brick house... he huffed, and he puffed..."
        " but he could not blow it down!"
        " The three little pigs lived happily ever after in the strong brick house."
    ),
    "goldilocks": (
        "[STORY] Once upon a time, there was a little girl named Goldilocks."
        " She went for a walk in the forest."
        " She came upon a house and knocked on the door. No one answered, so she walked right in!"
        " [HAPPY] At the table, there were three bowls of porridge."
        ' She tasted the first bowl. [sad] "This porridge is too hot!" she exclaimed.'
        ' She tasted the second bowl. [sad] "This porridge is too cold," she said.'
        ' She tasted the third bowl. [HAPPY] "Ahhh, this porridge is just right!" and she ate it all up.'
        " Then, the three bears came home!"
        ' [sad] "Someone\'s been eating my porridge!" growled the Papa Bear.'
        ' [sad] "Someone\'s been eating my porridge," said the Mama Bear.'
        ' [HAPPY] "Someone\'s been eating my porridge and they ate it all up!" cried the Baby Bear.'
        " Goldilocks woke up and ran away, never to return."
    ),
}


class StoryTime(Tool):
    """Tell a famous children's story with expressive narration."""

    name = "story_time"
    description = "Tell a famous children's story. Use this when the user asks for a story or to be entertained. Available stories: 'three_little_pigs', 'goldilocks'."
    parameters_schema = {
        "type": "object",
        "properties": {
            "story_name": {
                "type": "string",
                "enum": list(STORIES.keys()),
                "description": "The name of the story to tell.",
            },
        },
        "required": ["story_name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute the story telling."""
        story_name: str = kwargs.get("story_name", "")
        story_text = STORIES.get(story_name)
        if not story_text:
            # Fallback to random if invalid name (or LLM hallucination)
            story_name = random.choice(list(STORIES.keys()))
            story_text = STORIES[story_name]

        if deps.speak_func:
            # Clean up indentation
            clean_text = " ".join(story_text.split())
            await deps.speak_func(clean_text)
            return {"status": "success", "message": f"Told the story: {story_name}"}
        else:
            return {"status": "error", "message": "Speech function not available."}
