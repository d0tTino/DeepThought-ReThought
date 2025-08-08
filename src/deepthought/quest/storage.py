from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from ..services.db_manager import DBManager


@dataclass
class Evidence:
    id: Optional[int]
    objective_id: int
    content: str
    created: Optional[datetime] = None


@dataclass
class Epiphany:
    id: Optional[int]
    quest_id: int
    insight: str
    created: Optional[datetime] = None


@dataclass
class LieRecord:
    id: Optional[int]
    quest_id: int
    lie: str
    created: Optional[datetime] = None


@dataclass
class Objective:
    id: Optional[int]
    quest_id: int
    description: str
    status: str = "pending"
    created: Optional[datetime] = None
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Quest:
    id: Optional[int]
    name: str
    description: str
    status: str = "pending"
    created: Optional[datetime] = None
    objectives: List[Objective] = field(default_factory=list)
    epiphanies: List[Epiphany] = field(default_factory=list)
    lies: List[LieRecord] = field(default_factory=list)


class QuestStorage:
    """DAO layer for quests and related entities."""

    def __init__(self, db: DBManager) -> None:
        self.db = db

    async def add_quest(self, quest: Quest) -> int:
        await self.db.connect()
        assert self.db._db is not None
        cur = await self.db._db.execute(
            "INSERT INTO quests (name, description, status) VALUES (?, ?, ?)",
            (quest.name, quest.description, quest.status),
        )
        await self.db._db.commit()
        quest_id = cur.lastrowid
        quest.id = quest_id
        return quest_id

    async def add_objective(self, objective: Objective) -> int:
        await self.db.connect()
        assert self.db._db is not None
        cur = await self.db._db.execute(
            "INSERT INTO objectives (quest_id, description, status) VALUES (?, ?, ?)",
            (objective.quest_id, objective.description, objective.status),
        )
        await self.db._db.commit()
        obj_id = cur.lastrowid
        objective.id = obj_id
        return obj_id

    async def add_evidence(self, evidence: Evidence) -> int:
        await self.db.connect()
        assert self.db._db is not None
        cur = await self.db._db.execute(
            "INSERT INTO evidence (objective_id, content) VALUES (?, ?)",
            (evidence.objective_id, evidence.content),
        )
        await self.db._db.commit()
        evidence_id = cur.lastrowid
        evidence.id = evidence_id
        return evidence_id

    async def add_epiphany(self, epiphany: Epiphany) -> int:
        await self.db.connect()
        assert self.db._db is not None
        cur = await self.db._db.execute(
            "INSERT INTO epiphanies (quest_id, insight) VALUES (?, ?)",
            (epiphany.quest_id, epiphany.insight),
        )
        await self.db._db.commit()
        epiphany_id = cur.lastrowid
        epiphany.id = epiphany_id
        return epiphany_id

    async def add_lie(self, lie: LieRecord) -> int:
        await self.db.connect()
        assert self.db._db is not None
        cur = await self.db._db.execute(
            "INSERT INTO lie_ledger (quest_id, lie) VALUES (?, ?)",
            (lie.quest_id, lie.lie),
        )
        await self.db._db.commit()
        lie_id = cur.lastrowid
        lie.id = lie_id
        return lie_id

    async def get_quest(self, quest_id: int) -> Optional[Quest]:
        await self.db.connect()
        assert self.db._db is not None
        async with self.db._db.execute(
            "SELECT quest_id, name, description, status, created FROM quests WHERE quest_id=?",
            (quest_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        quest = Quest(
            id=row[0],
            name=row[1],
            description=row[2],
            status=row[3],
            created=datetime.fromisoformat(row[4]) if row[4] else None,
        )
        # objectives
        async with self.db._db.execute(
            "SELECT objective_id, description, status, created FROM objectives WHERE quest_id=?",
            (quest_id,),
        ) as cur:
            obj_rows = await cur.fetchall()
        for o in obj_rows:
            obj = Objective(
                id=o[0],
                quest_id=quest_id,
                description=o[1],
                status=o[2],
                created=datetime.fromisoformat(o[3]) if o[3] else None,
            )
            # evidence for objective
            async with self.db._db.execute(
                "SELECT evidence_id, content, created FROM evidence WHERE objective_id=?",
                (obj.id,),
            ) as ecur:
                e_rows = await ecur.fetchall()
            for e in e_rows:
                obj.evidence.append(
                    Evidence(
                        id=e[0],
                        objective_id=obj.id,
                        content=e[1],
                        created=datetime.fromisoformat(e[2]) if e[2] else None,
                    )
                )
            quest.objectives.append(obj)
        # epiphanies
        async with self.db._db.execute(
            "SELECT epiphany_id, insight, created FROM epiphanies WHERE quest_id=?",
            (quest_id,),
        ) as cur:
            epi_rows = await cur.fetchall()
        for e in epi_rows:
            quest.epiphanies.append(
                Epiphany(
                    id=e[0],
                    quest_id=quest_id,
                    insight=e[1],
                    created=datetime.fromisoformat(e[2]) if e[2] else None,
                )
            )
        # lies
        async with self.db._db.execute(
            "SELECT lie_id, lie, created FROM lie_ledger WHERE quest_id=?",
            (quest_id,),
        ) as cur:
            lie_rows = await cur.fetchall()
        for record in lie_rows:
            quest.lies.append(
                LieRecord(
                    id=record[0],
                    quest_id=quest_id,
                    lie=record[1],
                    created=datetime.fromisoformat(record[2]) if record[2] else None,
                )
            )
        return quest
