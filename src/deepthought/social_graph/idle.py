from __future__ import annotations

import asyncio
import os
from typing import Any

_idle_text_generator: Any | None = None


def _get_idle_generator():
    """Return a cached HuggingFace text-generation pipeline."""
    global _idle_text_generator
    if _idle_text_generator is None:
        from transformers import pipeline

        model_name = os.getenv("IDLE_MODEL_NAME", "distilgpt2")
        _idle_text_generator = pipeline("text-generation", model=model_name)
    return _idle_text_generator


async def generate_idle_response(prompt: str | None = None) -> str | None:
    """Generate a prompt to send when the channel has been idle."""
    try:
        gen_prompt = prompt or os.getenv(
            "IDLE_GENERATOR_PROMPT", "Say something to spark conversation."
        )

        generator = _get_idle_generator()
        outputs = await asyncio.to_thread(
            generator,
            gen_prompt,
            max_new_tokens=20,
            num_return_sequences=1,
        )

        text = outputs[0]["generated_text"].strip()
        return text
    except Exception:  # pragma: no cover - optional dependency or runtime error
        import logging

        logging.getLogger(__name__).exception("Idle text generation failed")
        return None


__all__ = ["generate_idle_response", "_get_idle_generator"]
