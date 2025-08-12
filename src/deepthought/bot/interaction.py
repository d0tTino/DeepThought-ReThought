"""Interaction helpers for the social graph bot."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Iterable

from deepthought.services.db_manager import DBManager

_db_manager_getter: Callable[[], DBManager] | None = None


def set_db_manager(getter: Callable[[], DBManager]) -> None:
    """Register a callable returning the active ``DBManager`` instance."""
    global _db_manager_getter
    _db_manager_getter = getter


def _db() -> DBManager:
    if _db_manager_getter is None:  # pragma: no cover - defensive
        raise RuntimeError("DB manager getter has not been set")
    return _db_manager_getter()


async def log_interaction(
    user_id: int,
    target_id: int | None = None,
    sentiment_score: float | None = None,
) -> None:
    await _db().log_interaction(user_id, target_id, sentiment_score=sentiment_score)


STYLE_LEVELS = ["silence", "emoji", "word", "one-liner", "paragraph"]


def choose_style(utility: float, tone: str = "neutral") -> str:
    """Select a reply style based on utility and tone preference.

    Parameters
    ----------
    utility:
        A score in ``[0, 1]`` indicating how useful a response would be.  Higher
        values correspond to longer replies.
    tone:
        User tone preference. ``"concise"`` biases toward shorter replies while
        ``"verbose"`` biases toward longer replies.

    Returns
    -------
    str
        One of ``"silence"``, ``"emoji"``, ``"word"``, ``"one-liner"``, or
        ``"paragraph"``.
    """

    idx = 0
    if utility >= 0.2:
        idx = 1
    if utility >= 0.4:
        idx = 2
    if utility >= 0.6:
        idx = 3
    if utility >= 0.8:
        idx = 4

    if tone == "concise":
        idx = max(0, idx - 1)
    elif tone == "verbose":
        idx = min(len(STYLE_LEVELS) - 1, idx + 1)

    return STYLE_LEVELS[idx]


def is_bot(user: object) -> bool:
    """Return ``True`` if a user object or name appears bot-like.

    The heuristic checks for a truthy ``bot`` attribute or the substring
    ``"bot"`` in the user's name.  It operates on minimal information to avoid
    depending on any specific chat platform.
    """

    if hasattr(user, "bot") and bool(getattr(user, "bot")):
        return True
    name = getattr(user, "name", str(user))
    return "bot" in name.lower()


recent_bot_messages: deque[tuple[datetime, str]] = deque(maxlen=20)


def record_bot_message(text: str) -> None:
    """Record a message sent by a bot for etiquette checks."""

    recent_bot_messages.append((datetime.utcnow(), text))


def is_crowded(participants: Iterable[object], bot_threshold: int = 2, *, window: float = 10.0) -> bool:
    """Return ``True`` if a conversation has too many bot participants or messages."""

    bot_count = sum(1 for p in participants if is_bot(p))
    if bot_count >= bot_threshold:
        return True

    cutoff = datetime.utcnow() - timedelta(seconds=window)
    while recent_bot_messages and recent_bot_messages[0][0] < cutoff:
        recent_bot_messages.popleft()
    return len(recent_bot_messages) >= bot_threshold


def novel_response(text: str, *, threshold: float = 0.5, window: float = 60.0) -> bool:
    """Return ``True`` if ``text`` is sufficiently different from recent bot messages."""

    cutoff = datetime.utcnow() - timedelta(seconds=window)
    tokens = set(text.lower().split())
    for ts, msg in list(recent_bot_messages):
        if ts < cutoff:
            continue
        other = set(msg.lower().split())
        if not tokens or not other:
            continue
        jaccard = len(tokens & other) / len(tokens | other)
        if jaccard >= threshold:
            return False
    return True
