"""Interaction helpers for the social graph bot."""

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


async def log_interaction(
    user_id: int,
    target_id: int | None = None,
    sentiment_score: float | None = None,
) -> None:
    await _db().log_interaction(user_id, target_id, sentiment_score=sentiment_score)
