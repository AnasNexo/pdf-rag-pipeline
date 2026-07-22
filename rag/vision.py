"""Image/chart/diagram description, rotating across vision providers.

Delegates the actual model call to rag.llm.describe_image, which fails over
across all configured vision-capable keys (Groq Llama-4, then Gemini). This
module only assembles the prompt and decides what to store when description
is impossible — never the raw API error, which would otherwise be indexed and
read back as if it were the image's content.
"""

from __future__ import annotations

from config import VISION_CONTEXT_MAX_CHARS, VISION_MAX_TOKENS
from rag.llm import describe_image as _describe_image_rotating

_DESCRIPTION_PROMPT = (
    "Describe this image in detail. If it is a chart, graph, or diagram, "
    "explain what data or process it represents, including key labels, "
    "trends, and values. If it is a photo or illustration, describe its "
    "content and relevance."
)

#: Stored in place of a description when every vision provider is exhausted.
#: Kept generic so it is never mistaken for actual image content at query time.
_UNAVAILABLE_PLACEHOLDER = "[image not described: vision service unavailable]"


def describe_image(image_bytes: bytes, context_text: str = "") -> str:
    """Generate a natural-language description of an image.

    Rotates across all configured vision providers (Groq then Gemini). If a
    description genuinely cannot be produced (all providers exhausted, or a
    hard error), returns a neutral placeholder rather than the error text, so
    the chunk store is never polluted with a 429/stack trace.

    Args:
        image_bytes (bytes): Encoded image data.
        context_text (str): Text from the surrounding page, truncated to
            VISION_CONTEXT_MAX_CHARS, to help disambiguate the image.

    Returns:
        str: Model-generated description, or a neutral placeholder on failure.
    """
    prompt = _DESCRIPTION_PROMPT
    if context_text:
        prompt += ("\n\nSurrounding page context:\n"
                   f"{context_text[:VISION_CONTEXT_MAX_CHARS]}")
    try:
        print("    [Step] Describing image (vision rotation)...")
        description = _describe_image_rotating(
            image_bytes, prompt, max_tokens=VISION_MAX_TOKENS)
        if not description:
            return _UNAVAILABLE_PLACEHOLDER
        print("    [Step] Vision description received")
        return description
    except Exception as e:  # noqa: BLE001 — store a neutral marker, not the error
        print(f"    [Step] Vision description unavailable: {e}")
        return _UNAVAILABLE_PLACEHOLDER
