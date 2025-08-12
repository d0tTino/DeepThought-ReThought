"""Utilities for sweeping expired quests.

This module provides an async task that inspects quests stored in a
``QuestStorage`` instance and moves any whose ``QuestFSM`` has expired to
``ABANDONED``.  A summary of moved quests is sent through
``QuestWriter.send_board_update``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Tuple

from .fsm import QuestFSM, QuestState
from .storage import Quest, QuestStorage
from .writer import QuestWriter


async def sweep_expired_quests(
    storage: QuestStorage,
    *,
    writer: QuestWriter | None = None,
    now: datetime | None = None,
) -> List[Quest]:
    """Transition expired quests and emit a digest of moved quests.

    Parameters
    ----------
    storage:
        Source of quests and their FSM metadata.  It must provide a
        ``list_quests`` coroutine returning an iterable of ``(Quest,
        QuestFSM)`` pairs.
    writer:
        Optional ``QuestWriter`` used to publish a summary of moved quests.
    now:
        Override the current time, primarily for testing.

    Returns
    -------
    list[Quest]
        Quests that were transitioned to ``ABANDONED``.
    """

    now = now or datetime.utcnow()
    moved: List[Quest] = []
    quests: Iterable[Tuple[Quest, QuestFSM]] = await storage.list_quests()

    for quest, fsm in quests:
        expiration = fsm.expires_at()
        if expiration and expiration <= now:
            before = fsm.state
            fsm.prune(now=now)
            if before != fsm.state and fsm.state == QuestState.ABANDONED:
                quest.status = QuestState.ABANDONED.value
                await storage.update_quest(quest)
                moved.append(quest)

    if moved and writer is not None:
        summary = {"abandoned": [q.id for q in moved]}
        writer.send_board_update(summary, event="sweeper")

    return moved
