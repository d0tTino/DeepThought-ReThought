from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from .storage import Epiphany, Evidence, LieRecord, Objective, Quest


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
            created=(datetime.fromisoformat(q["created"]) if q.get("created") else None),
            updated=(datetime.fromisoformat(q["updated"]) if q.get("updated") else None),
        )
        for o in q.get("objectives", []):
            obj = Objective(
                id=o.get("id"),
                quest_id=quest.id or 0,
                description=o["description"],
                status=o.get("status", "pending"),
                preconditions=o.get("preconditions", []),
                success_criteria=o.get("success_criteria", []),
                fail_criteria=o.get("fail_criteria", []),
                fallbacks=o.get("fallbacks", []),
                cooldowns=o.get("cooldowns", []),
            )
            for ev in o.get("evidence", []):
                if isinstance(ev, dict):
                    obj.evidence.append(
                        Evidence(
                            id=None,
                            objective_id=obj.id or 0,
                            content=ev.get("content", ""),
                            who=ev.get("who", ""),
                            confidence_delta=ev.get("confidence_delta", 0.0),
                            expiry=datetime.fromisoformat(ev["expiry"]) if ev.get("expiry") else None,
                        )
                    )
                else:
                    obj.evidence.append(Evidence(id=None, objective_id=obj.id or 0, content=ev))
            quest.objectives.append(obj)
        for epi in q.get("epiphanies", []):
            if isinstance(epi, dict):
                quest.epiphanies.append(
                    Epiphany(
                        id=None,
                        quest_id=quest.id or 0,
                        insight=epi.get("insight", ""),
                        who=epi.get("who", ""),
                        confidence_delta=epi.get("confidence_delta", 0.0),
                        expiry=datetime.fromisoformat(epi["expiry"]) if epi.get("expiry") else None,
                    )
                )
            else:
                quest.epiphanies.append(Epiphany(id=None, quest_id=quest.id or 0, insight=epi))
        for lie in q.get("lies", []):
            if isinstance(lie, dict):
                quest.lies.append(
                    LieRecord(
                        id=None,
                        quest_id=quest.id or 0,
                        lie=lie.get("lie", ""),
                        who=lie.get("who", ""),
                        confidence_delta=lie.get("confidence_delta", 0.0),
                        expiry=datetime.fromisoformat(lie["expiry"]) if lie.get("expiry") else None,
                    )
                )
            else:
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
                "preconditions": o.preconditions,
                "success_criteria": o.success_criteria,
                "fail_criteria": o.fail_criteria,
                "fallbacks": o.fallbacks,
                "cooldowns": o.cooldowns,
                "evidence": [
                    {
                        "content": e.content,
                        "who": e.who,
                        "confidence_delta": e.confidence_delta,
                        "expiry": e.expiry.isoformat() if e.expiry else None,
                    }
                    for e in o.evidence
                ],
            }
            for o in quest.objectives
        ],
        "epiphanies": [
            {
                "insight": e.insight,
                "who": e.who,
                "confidence_delta": e.confidence_delta,
                "expiry": e.expiry.isoformat() if e.expiry else None,
            }
            for e in quest.epiphanies
        ],
        "lies": [
            {
                "lie": lr.lie,
                "who": lr.who,
                "confidence_delta": lr.confidence_delta,
                "expiry": lr.expiry.isoformat() if lr.expiry else None,
            }
            for lr in quest.lies
        ],
    }
