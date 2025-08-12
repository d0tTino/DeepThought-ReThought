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

    The narrative includes the quest description and objectives along with
    cast lists, evidence summaries, unexpected twists and lessons learned.
    If ``writer`` is provided, narratives for completed quests are posted via
    ``QuestWriter.send_quest_story``.
    """

    narratives: List[str] = []
    for quest in quests:
        objectives = "; ".join(o.description for o in quest.objectives) or "no objectives"
        narrative = f"{quest.name}: {quest.description}. Objectives: {objectives}."

        cast = set()
        evidence_summaries: List[str] = []
        for obj in quest.objectives:
            for ev in obj.evidence:
                evidence_summaries.append(ev.content)
                if ev.who:
                    cast.add(ev.who)
        for epi in quest.epiphanies:
            if epi.who:
                cast.add(epi.who)
        for lie in quest.lies:
            if lie.who:
                cast.add(lie.who)

        if cast:
            narrative += " Cast: " + ", ".join(sorted(cast)) + "."
        if evidence_summaries:
            narrative += " Evidence: " + "; ".join(evidence_summaries) + "."
        if quest.lies:
            narrative += " Twists: " + "; ".join(lie.lie for lie in quest.lies) + "."
        if quest.epiphanies:
            narrative += " Lessons: " + "; ".join(e.insight for e in quest.epiphanies) + "."
        narratives.append(narrative)
        if writer:
            writer.send_quest_story(quest, narrative)
    return narratives


def generate_living_report(
    quests: Iterable[Quest],
    channel_activity: Dict[str, int],
    writer: QuestWriter | None = None,
) -> Dict[str, Any]:
    """Return a "living" report of weekly arcs and channel heatmap.

    ``channel_activity`` should map channel names to message counts. When a
    ``writer`` is supplied the report is dispatched via
    :meth:`QuestWriter.send_living_report`.
    """

    arcs: Dict[str, List[str]] = {}
    for quest in quests:
        timestamp = quest.updated or quest.created or datetime.utcnow()
        week_start = (timestamp - timedelta(days=timestamp.weekday())).date().isoformat()
        arcs.setdefault(week_start, []).append(f"{quest.name}: {quest.status}")

    weekly_arcs = [
        {"week": week, "summary": "; ".join(entries)} for week, entries in sorted(arcs.items())
    ]
    report = {"weekly_arcs": weekly_arcs, "channel_heatmap": channel_activity}
    if writer:
        writer.send_living_report(report)
    return report


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
