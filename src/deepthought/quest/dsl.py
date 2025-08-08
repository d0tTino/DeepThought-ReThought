from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from .storage import Quest, Objective, Evidence, Epiphany, LieRecord


def load_quests(path: str | Path) -> List[Quest]:
    """Load quests from a JSON DSL file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    quests: List[Quest] = []
    for q in data.get("quests", []):
        quest = Quest(
            id=q.get("id"),
            name=q["name"],
            description=q.get("description", ""),
            quest_type=q.get("quest_type", ""),
            priority=q.get("priority", 0),
            horizon=q.get("horizon", ""),
            faction=q.get("faction", ""),
            cover_story=q.get("cover_story", ""),
            secrecy=q.get("secrecy", ""),
            risk=q.get("risk", ""),
            status=q.get("status", "pending"),
            created=(
                datetime.fromisoformat(q["created"]) if q.get("created") else None
            ),
            updated=(
                datetime.fromisoformat(q["updated"]) if q.get("updated") else None
            ),
        )
        for o in q.get("objectives", []):
            obj = Objective(
                id=o.get("id"),
                quest_id=quest.id or 0,
                description=o["description"],
                status=o.get("status", "pending"),
            )
            for ev in o.get("evidence", []):
                obj.evidence.append(Evidence(id=None, objective_id=obj.id or 0, content=ev))
            quest.objectives.append(obj)
        for epi in q.get("epiphanies", []):
            quest.epiphanies.append(Epiphany(id=None, quest_id=quest.id or 0, insight=epi))
        for lie in q.get("lies", []):
            quest.lies.append(LieRecord(id=None, quest_id=quest.id or 0, lie=lie))
        quests.append(quest)
    return quests


def save_quests(path: str | Path, quests: List[Quest]) -> None:
    """Save quests to a JSON DSL file."""
    data = {"quests": [quest_to_dict(q) for q in quests]}
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def quest_to_dict(quest: Quest) -> dict:
    return {
        "id": quest.id,
        "name": quest.name,
        "description": quest.description,
        "quest_type": quest.quest_type,
        "priority": quest.priority,
        "horizon": quest.horizon,
        "faction": quest.faction,
        "cover_story": quest.cover_story,
        "secrecy": quest.secrecy,
        "risk": quest.risk,
        "status": quest.status,
        "created": quest.created.isoformat() if quest.created else None,
        "updated": quest.updated.isoformat() if quest.updated else None,
        "objectives": [
            {
                "id": o.id,
                "description": o.description,
                "status": o.status,
                "evidence": [e.content for e in o.evidence],
            }
            for o in quest.objectives
        ],
        "epiphanies": [e.insight for e in quest.epiphanies],
        "lies": [lr.lie for lr in quest.lies],
    }
