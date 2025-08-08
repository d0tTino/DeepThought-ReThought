from __future__ import annotations

"""Quest template definitions and helper utilities.

This module introduces a small quest templating system. Each template
represents a commonly recurring quest pattern (Main story line, side quest,
self-care reminders, etc.). Templates specify a default cooldown period and
associate with a *horizon* budget which limits how many quests of that type can
exist simultaneously.

Two helpers are provided:

``bind_slot``
    Bind a quest template to a user from the social graph while respecting a
    cooldown tracker.
``auto_spawn_quests``
    Spawn quests automatically when budget allows. The function returns a list
    of ``Quest`` instances ready for persistence via :class:`QuestStorage`.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

from .storage import Quest
from ..services.social_graph_memory import SocialGraphMemory


@dataclass
class QuestTemplate:
    """Definition of a quest template."""

    name: str
    horizon: str
    description: str = ""
    cooldown: timedelta = timedelta(hours=1)


# Template definitions -----------------------------------------------------
MAIN = QuestTemplate("Main", "long", "Primary narrative arc")
INVESTIGATION = QuestTemplate("Investigation", "medium", "Follow up on a clue")
SIDE = QuestTemplate("Side", "short", "Optional or flavour quest")
EXPERIMENT = QuestTemplate("Experiment", "short", "Try something novel")
SELF_CARE = QuestTemplate("Self-Care", "short", "Look after the agent")
OPS_SEC = QuestTemplate("OpsSec", "medium", "Operational security checks")
BRIDGE_BUILDER = QuestTemplate("Bridge-Builder", "medium", "Improve relations between users")
RED_TEAM = QuestTemplate("Red Team", "long", "Adversarial challenge")

TEMPLATES: Tuple[QuestTemplate, ...] = (
    MAIN,
    INVESTIGATION,
    SIDE,
    EXPERIMENT,
    SELF_CARE,
    OPS_SEC,
    BRIDGE_BUILDER,
    RED_TEAM,
)


class CooldownTracker:
    """In-memory tracker storing last assignment timestamps."""

    def __init__(self) -> None:
        self._last: Dict[Tuple[str, str], datetime] = {}

    def ready(self, user: str, template: QuestTemplate, *, now: Optional[datetime] = None) -> bool:
        now = now or datetime.utcnow()
        last = self._last.get((user, template.name))
        return last is None or now - last >= template.cooldown

    def mark(self, user: str, template: QuestTemplate, *, now: Optional[datetime] = None) -> None:
        self._last[(user, template.name)] = now or datetime.utcnow()


async def _list_users(memory: SocialGraphMemory) -> List[str]:
    """Return all known users from the social graph."""

    await memory._db.connect()
    assert memory._db._db is not None  # for type checkers
    async with memory._db._db.execute("SELECT DISTINCT user_id FROM affinity") as cur:
        rows = await cur.fetchall()
    return [r[0] for r in rows]


async def bind_slot(
    template: QuestTemplate,
    memory: SocialGraphMemory,
    tracker: CooldownTracker,
) -> Optional[str]:
    """Bind ``template`` to the best available user slot.

    The user with the highest affinity score not currently on cooldown is
    selected. Returns ``None`` when no suitable user is found.
    """

    users = await _list_users(memory)
    best_user: Optional[str] = None
    best_score: float = float("-inf")
    for uid in users:
        if not tracker.ready(uid, template):
            continue
        score = await memory.get_affinity(uid)
        if score > best_score:
            best_user = uid
            best_score = score
    if best_user is not None:
        tracker.mark(best_user, template)
    return best_user


async def auto_spawn_quests(
    budget: Dict[str, int],
    memory: SocialGraphMemory,
    tracker: CooldownTracker,
    templates: Iterable[QuestTemplate] = TEMPLATES,
) -> List[Quest]:
    """Spawn quests automatically based on ``budget``.

    ``budget`` maps horizon names to remaining spawn counts. For each template
    we attempt to bind a slot and, if the horizon's budget permits, create a
    ``Quest`` instance.
    """

    spawned: List[Quest] = []
    for tmpl in templates:
        remaining = budget.get(tmpl.horizon, 0)
        if remaining <= 0:
            continue
        slot = await bind_slot(tmpl, memory, tracker)
        if slot is None:
            continue
        quest = Quest(id=None, name=tmpl.name, description=tmpl.description)
        spawned.append(quest)
        budget[tmpl.horizon] = remaining - 1
    return spawned
