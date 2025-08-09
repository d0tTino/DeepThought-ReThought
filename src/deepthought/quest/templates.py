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

Auto spawning further enforces per-horizon budgets with cooldown and TTL rules
through :class:`HorizonManager`.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

from ..services.social_graph_memory import SocialGraphMemory
from .storage import Quest


@dataclass
class HorizonRule:
    """Configuration for a quest horizon.

    ``limit``
        Maximum concurrent quests allowed for the horizon.
    ``cooldown``
        Minimum time that must elapse between quest spawns for the horizon.
    ``ttl``
        Time-to-live for quests; once expired they free up budget again.
    """

    limit: int
    cooldown: timedelta = timedelta(0)
    ttl: timedelta = timedelta(hours=1)


class HorizonManager:
    """Track active quests per horizon enforcing budget, cooldown and TTL."""

    def __init__(self, rules: Dict[str, HorizonRule]) -> None:
        self.rules = rules
        self._active: Dict[str, List[datetime]] = {h: [] for h in rules}
        self._last_spawn: Dict[str, datetime] = {}

    def _cleanup(self, horizon: str, now: datetime) -> None:
        rule = self.rules[horizon]
        self._active[horizon] = [t for t in self._active[horizon] if now - t < rule.ttl]

    def can_spawn(self, horizon: str, *, now: Optional[datetime] = None) -> bool:
        now = now or datetime.utcnow()
        if horizon not in self.rules:
            return False
        self._cleanup(horizon, now)
        rule = self.rules[horizon]
        if len(self._active[horizon]) >= rule.limit:
            return False
        last = self._last_spawn.get(horizon)
        if last and now - last < rule.cooldown:
            return False
        return True

    def record_spawn(self, horizon: str, *, now: Optional[datetime] = None) -> None:
        now = now or datetime.utcnow()
        self._cleanup(horizon, now)
        self._active[horizon].append(now)
        self._last_spawn[horizon] = now


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


async def _find_low_affinity_pair(memory: SocialGraphMemory) -> Optional[Tuple[str, str]]:
    """Return the pair of users with the lowest mutual affinity.

    This uses :class:`SocialGraphMemory` to inspect relationship metrics and
    identify users most in need of a bridge-building quest.
    """

    users = await _list_users(memory)
    if len(users) < 2:
        return None
    worst_pair: Optional[Tuple[str, str]] = None
    worst_score = float("inf")
    for i, a in enumerate(users):
        start = i + 1
        for b in users[start:]:
            score = await memory.get_mutual_affinity(a, b)
            if score < worst_score:
                worst_score = score
                worst_pair = (a, b)
    return worst_pair


async def auto_spawn_quests(
    manager: HorizonManager,
    memory: SocialGraphMemory,
    tracker: CooldownTracker,
    templates: Iterable[QuestTemplate] = TEMPLATES,
    *,
    now: Optional[datetime] = None,
) -> List[Quest]:
    """Spawn quests automatically based on horizon rules.

    ``manager`` encapsulates horizon budgets with cooldown and TTL enforcement.
    For each template we attempt to bind a slot (or pair) and, if allowed by the
    horizon, create a ``Quest`` instance.
    """

    current = now or datetime.utcnow()
    spawned: List[Quest] = []
    for tmpl in templates:
        if not manager.can_spawn(tmpl.horizon, now=current):
            continue
        if tmpl is BRIDGE_BUILDER:
            pair = await _find_low_affinity_pair(memory)
            if pair is None:
                continue
            quest = Quest(
                id=None,
                name=tmpl.name,
                description=f"{tmpl.description}: {pair[0]} vs {pair[1]}",
                horizon=tmpl.horizon,
            )
        else:
            slot = await bind_slot(tmpl, memory, tracker)
            if slot is None:
                continue
            quest = Quest(
                id=None,
                name=tmpl.name,
                description=tmpl.description,
                horizon=tmpl.horizon,
            )
        spawned.append(quest)
        manager.record_spawn(tmpl.horizon, now=current)
    return spawned
