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
    updated: Optional[datetime] = None


@dataclass
class Epiphany:
    id: Optional[int]
    quest_id: int
    insight: str
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


@dataclass
class LieRecord:
    id: Optional[int]
    quest_id: int
    lie: str
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


@dataclass
class Objective:
    id: Optional[int]
    quest_id: int
    description: str
    status: str = "pending"
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Quest:
    id: Optional[int]
    name: str
    description: str
    quest_type: str = ""
    priority: int = 0
    horizon: str = ""
    faction: str = ""
    cover_story: str = ""
    secrecy: str = ""
    risk: str = ""
    status: str = "pending"
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
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
            """
            INSERT INTO quests (
                name, description, quest_type, priority, horizon, faction,
                cover_story, secrecy, risk, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                quest.name,
                quest.description,
                quest.quest_type,
                quest.priority,
                quest.horizon,
                quest.faction,
                quest.cover_story,
                quest.secrecy,
                quest.risk,
                quest.status,
            ),
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
            """
            SELECT quest_id, name, description, quest_type, priority, horizon,
                   faction, cover_story, secrecy, risk, status, created, updated
            FROM quests WHERE quest_id=?
            """,
            (quest_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        quest = Quest(
            id=row[0],
            name=row[1],
            description=row[2],
            quest_type=row[3] or "",
            priority=row[4] or 0,
            horizon=row[5] or "",
            faction=row[6] or "",
            cover_story=row[7] or "",
            secrecy=row[8] or "",
            risk=row[9] or "",
            status=row[10],
            created=datetime.fromisoformat(row[11]) if row[11] else None,
            updated=datetime.fromisoformat(row[12]) if row[12] else None,
        )
        # objectives
        async with self.db._db.execute(
            "SELECT objective_id, description, status, created, updated FROM objectives WHERE quest_id=?",
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
                updated=datetime.fromisoformat(o[4]) if o[4] else None,
            )
            # evidence for objective
            async with self.db._db.execute(
                "SELECT evidence_id, content, created, updated FROM evidence WHERE objective_id=?",
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
                        updated=datetime.fromisoformat(e[3]) if e[3] else None,
                    )
                )
            quest.objectives.append(obj)
        # epiphanies
        async with self.db._db.execute(
            "SELECT epiphany_id, insight, created, updated FROM epiphanies WHERE quest_id=?",
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
                    updated=datetime.fromisoformat(e[3]) if e[3] else None,
                )
            )
        # lies
        async with self.db._db.execute(
            "SELECT lie_id, lie, created, updated FROM lie_ledger WHERE quest_id=?",
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
                    updated=datetime.fromisoformat(record[3]) if record[3] else None,
                )
            )
        return quest

    async def update_quest(self, quest: Quest) -> None:
        """Persist changes to a quest and update the timestamp."""
        if quest.id is None:
            raise ValueError("quest.id is required for update")
        await self.db.connect()
        assert self.db._db is not None
        await self.db._db.execute(
            """
            UPDATE quests SET
                name=?, description=?, quest_type=?, priority=?, horizon=?,
                faction=?, cover_story=?, secrecy=?, risk=?, status=?,
                updated=CURRENT_TIMESTAMP
            WHERE quest_id=?
            """,
            (
                quest.name,
                quest.description,
                quest.quest_type,
                quest.priority,
                quest.horizon,
                quest.faction,
                quest.cover_story,
                quest.secrecy,
                quest.risk,
                quest.status,
                quest.id,
            ),
        )
        await self.db._db.commit()

    async def delete_quest(self, quest_id: int) -> None:
        """Delete a quest and its related records."""
        await self.db.connect()
        assert self.db._db is not None
        # Delete child records first to maintain integrity
        await self.db._db.execute("DELETE FROM objectives WHERE quest_id=?", (quest_id,))
        await self.db._db.execute("DELETE FROM epiphanies WHERE quest_id=?", (quest_id,))
        await self.db._db.execute("DELETE FROM lie_ledger WHERE quest_id=?", (quest_id,))
        await self.db._db.execute("DELETE FROM quests WHERE quest_id=?", (quest_id,))
        await self.db._db.commit()
