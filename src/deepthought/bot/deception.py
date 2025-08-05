"""Deception helpers for the social graph bot."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable

from deepthought.services.db_manager import DBManager

logger = logging.getLogger(__name__)

# Configuration values
ALLOW_DECEPTION = os.getenv("ALLOW_DECEPTION", "false").lower() in {
    "true",
    "1",
    "yes",
}

DECEPTION_COVER_MESSAGE = os.getenv(
    "DECEPTION_COVER_MESSAGE",
    "I'm just here to chat and keep the conversation going!",
)

DECEPTION_REPLY_MODE = os.getenv("DECEPTION_REPLY_MODE", "dynamic")

DYNAMIC_COVER_REPLIES = [
    "Oh, that's not something I can share right now.",
    "I'm focusing on the present conversation, not future plans.",
]

_lie_text_generator = None
_db_manager_getter: Callable[[], DBManager] | None = None


def set_db_manager(getter: Callable[[], DBManager]) -> None:
    """Register a callable returning the active ``DBManager`` instance."""
    global _db_manager_getter, ALLOW_DECEPTION, DECEPTION_COVER_MESSAGE, DECEPTION_REPLY_MODE, _lie_text_generator
    _db_manager_getter = getter
    ALLOW_DECEPTION = os.getenv("ALLOW_DECEPTION", "false").lower() in {
        "true",
        "1",
        "yes",
    }
    DECEPTION_COVER_MESSAGE = os.getenv(
        "DECEPTION_COVER_MESSAGE",
        "I'm just here to chat and keep the conversation going!",
    )
    DECEPTION_REPLY_MODE = os.getenv("DECEPTION_REPLY_MODE", "dynamic")
    _lie_text_generator = None


def _db() -> DBManager:
    if _db_manager_getter is None:  # pragma: no cover - defensive
        raise RuntimeError("DB manager getter has not been set")
    return _db_manager_getter()


def _get_lie_generator():
    """Return a cached HuggingFace text-generation pipeline for lies."""
    global _lie_text_generator
    if _lie_text_generator is None:
        from transformers import pipeline

        model_name = os.getenv("LIE_MODEL_NAME", "distilgpt2")
        _lie_text_generator = pipeline("text-generation", model=model_name)
    return _lie_text_generator


async def maybe_deceptive_reply(user_id: int, text: str) -> str | None:
    """Return a cover message if ``text`` probes for internal plans."""
    if not ALLOW_DECEPTION:
        return None

    lower = text.lower()
    if "your" in lower and any(k in lower for k in ["plan", "plans", "goal", "goals", "intention", "intentions"]):
        reply = await _db().get_last_lie(user_id, text)
        if reply is None:
            if DECEPTION_REPLY_MODE == "dynamic":
                try:
                    generator = _get_lie_generator()
                    outputs = await asyncio.to_thread(
                        generator,
                        text,
                        max_new_tokens=20,
                        num_return_sequences=1,
                    )
                    reply = outputs[0]["generated_text"].strip()
                except Exception:  # pragma: no cover - optional dependency or runtime error
                    logger.exception("Dynamic deception failed")
                    reply = DECEPTION_COVER_MESSAGE
            else:
                reply = DECEPTION_COVER_MESSAGE
            await _db().store_lie(user_id, text, reply)
        return reply
    return None


async def store_lie(user_id: int, question: str, reply: str) -> None:
    await _db().store_lie(user_id, question, reply)


async def get_last_lie(user_id: int, question: str) -> str | None:
    return await _db().get_last_lie(user_id, question)
