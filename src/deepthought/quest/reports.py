from __future__ import annotations

"""Utilities for generating quest progress reports.

This module provides helpers to assemble narrative summaries, track weekly
faction momentum and produce case file dictionaries suitable for archival or
transmission.  It also includes a small scheduler that periodically sends the
compiled summary to the "Thought Server" using the ``DiscordThoughtWriter``
from :mod:`deepthought.planning.stacked_planner`.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    from ..planning.stacked_planner import DiscordThoughtWriter
except Exception:  # pragma: no cover

    class DiscordThoughtWriter:  # type: ignore[no-redef]
        def send(self, summary: dict) -> None:  # pragma: no cover - placeholder
            """Fallback writer used when planner module is unavailable."""
            pass


from .storage import Quest
from .writer import QuestWriter

# ---------------------------------------------------------------------------
# Report compilation helpers


def compile_narratives(quests: Iterable[Quest], writer: QuestWriter | None = None) -> List[str]:
    """Return human readable narratives for each quest.

    The narrative includes the quest description, objectives, epiphanies and
    notable lies recorded along the way. If ``writer`` is provided, narratives
    for completed quests are posted via ``QuestWriter.send_quest_story``.
    """

    narratives: List[str] = []
    for quest in quests:
        objectives = "; ".join(o.description for o in quest.objectives) or "no objectives"
        narrative = f"{quest.name}: {quest.description}. Objectives: {objectives}."
        if quest.epiphanies:
            narrative += " Epiphanies: " + "; ".join(e.insight for e in quest.epiphanies) + "."
        if quest.lies:
            narrative += " Lies: " + "; ".join(lie.lie for lie in quest.lies) + "."
        narratives.append(narrative)
        if writer:
            writer.send_quest_story(quest, narrative)
    return narratives


def weekly_faction_shifts(quests: Iterable[Quest]) -> Dict[str, int]:
    """Compute weekly faction momentum.

    Completed quests count as +1 for their faction while failed quests count as
    -1. Other statuses do not influence the shift.
    """

    shifts: Dict[str, int] = {}
    for quest in quests:
        if not quest.faction:
            continue
        delta = 1 if quest.status == "completed" else -1 if quest.status == "failed" else 0
        if delta:
            shifts[quest.faction] = shifts.get(quest.faction, 0) + delta
    return shifts


def case_files(quests: Iterable[Quest]) -> List[Dict[str, Any]]:
    """Produce serialisable case file dictionaries for each quest."""

    files: List[Dict[str, Any]] = []
    for quest in quests:
        files.append(
            {
                "id": quest.id,
                "name": quest.name,
                "status": quest.status,
                "objectives": [o.description for o in quest.objectives],
                "epiphanies": [e.insight for e in quest.epiphanies],
                "lies": [lie.lie for lie in quest.lies],
            }
        )
    return files


# ---------------------------------------------------------------------------
# Scheduling logic


@dataclass
class SummaryScheduler:
    """Send periodic quest summaries to the Thought Server."""

    interval: timedelta = timedelta(days=7)
    writer: DiscordThoughtWriter | None = None
    _next_run: datetime = datetime.min

    def maybe_send(self, quests: Iterable[Quest], now: datetime | None = None) -> None:
        """Send a compiled summary if the scheduled interval has passed."""

        now = now or datetime.utcnow()
        if now < self._next_run:
            return
        summary = {
            "narratives": compile_narratives(quests),
            "faction_shifts": weekly_faction_shifts(quests),
            "case_files": case_files(quests),
        }
        writer = self.writer or DiscordThoughtWriter()
        writer.send(summary)
        self._next_run = now + self.interval
