"""Memory helper functions for the social graph bot."""

from __future__ import annotations

from typing import Callable

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


async def recall_user(user_id: int, limit: int | None = None):
    return await _db().recall_user(user_id, limit=limit)


async def store_memory(
    user_id: int,
    memory: str,
    topic: str = "",
    sentiment_score: float | None = None,
) -> None:
    await _db().store_memory(user_id, memory, topic=topic, sentiment_score=sentiment_score)


async def store_theory(subject_id: int, theory: str, confidence: float) -> None:
    return await _db().store_theory(subject_id, theory, confidence)


async def get_theories(subject_id: int):
    return await _db().get_theories(subject_id)


async def update_sentiment_trend(
    user_id: int,
    channel_id: int,
    sentiment_score: float,
) -> None:
    await _db().update_sentiment_trend(user_id, channel_id, sentiment_score)


async def get_sentiment_trend(user_id: int, channel_id: int):
    return await _db().get_sentiment_trend(user_id, channel_id)


async def get_recent_topics(limit: int = 3) -> list[str]:
    return await _db().get_recent_topics(limit)


async def queue_deep_reflection(user_id: int, context: dict, prompt: str) -> int:
    return await _db().queue_deep_reflection(user_id, context, prompt)


async def add_summary_goal(user_id: int, context: dict, prompt: str) -> int:
    return await _db().add_summary_goal(user_id, context, prompt)


async def list_pending_summary_goals():
    return await _db().list_pending_summary_goals()


async def mark_summary_goal_done(task_id: int) -> None:
    await _db().mark_summary_goal_done(task_id)


async def set_do_not_mock(user_id: int, flag: bool = True) -> None:
    await _db().set_do_not_mock(user_id, flag)


async def is_do_not_mock(user_id: int) -> bool:
    return await _db().is_do_not_mock(user_id)


async def adjust_affinity(user_id: int, delta: float) -> None:
    await _db().adjust_affinity(user_id, delta)


async def get_affinity(user_id: int) -> int:
    return await _db().get_affinity(user_id)


async def get_friendliness(user_id: int, target_id: int) -> float:
    return await _db().get_friendliness(user_id, target_id)


async def get_hostility(user_id: int, target_id: int) -> float:
    return await _db().get_hostility(user_id, target_id)


async def get_interaction_weight(user_id: int, target_id: int) -> float:
    return await _db().get_interaction_weight(user_id, target_id)


async def get_last_interaction(user_id: int, target_id: int):
    return await _db().get_last_interaction(user_id, target_id)


async def get_pair_mutual_affinity(user_a: int, user_b: int) -> float:
    return await _db().get_pair_mutual_affinity(user_a, user_b)


async def set_theme(user_id: int, channel_id: int, theme: str) -> None:
    await _db().set_theme(user_id, channel_id, theme)


async def get_theme(user_id: int, channel_id: int):
    """Return the last assigned theme for a user/channel pair."""
    return await _db().get_theme(user_id, channel_id)


async def assign_themes() -> None:
    """Update the theme for each user/channel based on sentiment trends."""
    rows = await _db().get_all_sentiment_trends()
    for user_id, channel_id, ssum, count in rows:
        if not count:
            continue
        avg = ssum / count
        if avg > 0.2:
            theme = "positive"
        elif avg < -0.2:
            theme = "negative"
        else:
            theme = "neutral"
        await _db().set_theme(user_id, channel_id, theme)
