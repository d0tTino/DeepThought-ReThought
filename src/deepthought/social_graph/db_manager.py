from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import aiosqlite

from .constants import (CURRENT_DB_PATH, DB_PATH, MAX_MEMORY_LENGTH,
                        MAX_PROMPT_LENGTH, MAX_THEORY_LENGTH)


class DBManager:
    """Lightweight wrapper managing a single aiosqlite connection."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._db is None:
            dir_path = os.path.dirname(self.db_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            self._db = await aiosqlite.connect(self.db_path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def init_db(self) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                user_id TEXT,
                target_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS affinity (
                user_id TEXT PRIMARY KEY,
                score INTEGER DEFAULT 0
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                user_id TEXT,
                topic TEXT,
                memory TEXT,
                sentiment_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS theories (
                subject_id TEXT,
                theory TEXT,
                confidence REAL,
                updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(subject_id, theory)
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS queued_tasks (
                task_id INTEGER PRIMARY KEY,
                user_id TEXT,
                context TEXT,
                prompt TEXT,
                status TEXT DEFAULT 'pending',
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiment_trends (
                user_id TEXT,
                channel_id TEXT,
                sentiment_sum REAL DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                PRIMARY KEY(user_id, channel_id)

            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS themes (
                user_id TEXT,
                channel_id TEXT,
                theme TEXT,
                updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, channel_id)
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_flags (
                user_id TEXT PRIMARY KEY,
                do_not_mock INTEGER
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS recent_topics (
                topic TEXT PRIMARY KEY,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP

            )
            """
        )
        await self._db.commit()

    async def log_interaction(self, user_id: int, target_id: int) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            "INSERT INTO interactions (user_id, target_id) VALUES (?, ?)",
            (str(user_id), str(target_id)),
        )
        await self._db.execute(
            """
            INSERT INTO affinity (user_id, score)
            VALUES (?, 1)
            ON CONFLICT(user_id) DO UPDATE SET score=affinity.score + 1
            """,
            (str(user_id),),
        )
        await self._db.commit()

    async def recall_user(self, user_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT topic, memory FROM memories WHERE user_id= ?",
            (str(user_id),),
        ) as cur:
            return await cur.fetchall()

    async def store_memory(
        self,
        user_id: int,
        memory: str,
        topic: str = "",
        sentiment_score: float | None = None,
    ) -> None:
        if not isinstance(memory, str) or not memory.strip():
            raise ValueError("memory must be a non-empty string")
        if len(memory) > MAX_MEMORY_LENGTH:
            raise ValueError("memory exceeds maximum length")
        if not isinstance(topic, str):
            raise ValueError("topic must be a string")
        if sentiment_score is not None:
            if not isinstance(sentiment_score, (int, float)):
                raise ValueError("sentiment_score must be numeric")
            if not -1 <= float(sentiment_score) <= 1:
                raise ValueError("sentiment_score out of range")

        await self.connect()
        assert self._db
        await self._db.execute(
            "INSERT INTO memories (user_id, topic, memory, sentiment_score) VALUES (?, ?, ?, ?)",
            (str(user_id), topic, memory, sentiment_score),
        )
        if topic:
            await self._db.execute(
                """
                INSERT INTO recent_topics (topic, last_used)
                VALUES (?, CURRENT_TIMESTAMP)
                ON CONFLICT(topic) DO UPDATE SET last_used=CURRENT_TIMESTAMP
                """,
                (topic,),
            )
        await self._db.commit()

    async def store_theory(
        self, subject_id: int, theory: str, confidence: float
    ) -> None:
        if not isinstance(theory, str) or not theory.strip():
            raise ValueError("theory must be a non-empty string")
        if len(theory) > MAX_THEORY_LENGTH:
            raise ValueError("theory exceeds maximum length")
        if not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("confidence out of range")

        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO theories (subject_id, theory, confidence)
            VALUES (?, ?, ?)
            ON CONFLICT(subject_id, theory) DO UPDATE SET
                confidence=excluded.confidence,
                updated=CURRENT_TIMESTAMP
            """,
            (str(subject_id), theory, confidence),
        )
        await self._db.commit()

    async def get_theories(self, subject_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT theory, confidence FROM theories WHERE subject_id=?",
            (str(subject_id),),
        ) as cur:
            return await cur.fetchall()

    async def update_sentiment_trend(
        self,
        user_id: int,
        channel_id: int,
        sentiment_score: float,
    ) -> None:
        if not isinstance(sentiment_score, (int, float)):
            raise ValueError("sentiment_score must be numeric")
        if not -1 <= float(sentiment_score) <= 1:
            raise ValueError("sentiment_score out of range")
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO sentiment_trends (user_id, channel_id, sentiment_sum, message_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(user_id, channel_id) DO UPDATE SET
                sentiment_sum=sentiment_trends.sentiment_sum + excluded.sentiment_sum,
                message_count=sentiment_trends.message_count + 1
            """,
            (str(user_id), str(channel_id), sentiment_score),
        )
        await self._db.commit()

    async def get_sentiment_trend(self, user_id: int, channel_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT sentiment_sum, message_count FROM sentiment_trends WHERE user_id=? AND channel_id=?",
            (str(user_id), str(channel_id)),
        ) as cur:
            return await cur.fetchone()

    async def get_recent_topics(self, limit: int = 3) -> list[str]:
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT topic FROM recent_topics ORDER BY last_used DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
            return [r[0] for r in rows]

    async def queue_deep_reflection(
        self, user_id: int, context: dict, prompt: str
    ) -> int:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError("prompt exceeds maximum length")
        if not isinstance(context, dict):
            raise ValueError("context must be a dictionary")
        try:
            context_json = json.dumps(context)
        except (TypeError, ValueError) as exc:
            raise ValueError("context is not JSON serializable") from exc

        await self.connect()
        assert self._db
        cur = await self._db.execute(
            "INSERT INTO queued_tasks (user_id, context, prompt) VALUES (?, ?, ?)",
            (str(user_id), context_json, prompt),
        )
        await self._db.commit()
        return cur.lastrowid

    async def list_pending_tasks(self):
        """Return pending reflection tasks."""
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT task_id, user_id, context, prompt FROM queued_tasks WHERE status='pending'"
        ) as cur:
            return await cur.fetchall()

    async def mark_task_done(self, task_id: int) -> None:
        """Mark a queued task as completed."""
        await self.connect()
        assert self._db
        await self._db.execute(
            "UPDATE queued_tasks SET status='done' WHERE task_id=?",
            (task_id,),
        )
        await self._db.commit()

    async def set_do_not_mock(self, user_id: int, flag: bool = True) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO user_flags (user_id, do_not_mock)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET do_not_mock=excluded.do_not_mock
            """,
            (str(user_id), int(flag)),
        )
        await self._db.commit()

    async def is_do_not_mock(self, user_id: int) -> bool:
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT do_not_mock FROM user_flags WHERE user_id=?",
            (str(user_id),),
        ) as cur:
            row = await cur.fetchone()
            return bool(row[0]) if row else False

    async def adjust_affinity(self, user_id: int, delta: int) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO affinity (user_id, score)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET score=affinity.score + ?
            """,
            (str(user_id), delta, delta),
        )
        await self._db.commit()

    async def get_affinity(self, user_id: int) -> int:
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT score FROM affinity WHERE user_id=?",
            (str(user_id),),
        ) as cur:
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def set_theme(self, user_id: int, channel_id: int, theme: str) -> None:
        if not isinstance(theme, str) or not theme.strip():
            raise ValueError("theme must be a non-empty string")
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO themes (user_id, channel_id, theme)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, channel_id) DO UPDATE SET
                theme=excluded.theme,
                updated=CURRENT_TIMESTAMP
            """,
            (str(user_id), str(channel_id), theme),
        )
        await self._db.commit()

    async def get_theme(self, user_id: int, channel_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT theme FROM themes WHERE user_id=? AND channel_id=?",
            (str(user_id), str(channel_id)),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else None

    async def get_all_sentiment_trends(self):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT user_id, channel_id, sentiment_sum, message_count FROM sentiment_trends"
        ) as cur:
            return await cur.fetchall()


db_manager = DBManager()


async def init_db(db_path: str | None = None) -> None:
    """Initialize the database, recreating the manager when the path changes."""
    global db_manager, CURRENT_DB_PATH

    target_path = (
        db_path
        if db_path is not None
        else (
            DB_PATH
            if DB_PATH != CURRENT_DB_PATH and db_manager.db_path == CURRENT_DB_PATH
            else db_manager.db_path
        )
    )

    if db_manager.db_path != target_path:
        if db_manager._db is not None:
            await db_manager.close()
        db_manager = DBManager(target_path)

    await db_manager.init_db()
    CURRENT_DB_PATH = db_manager.db_path


async def log_interaction(user_id: int, target_id: int) -> None:
    await db_manager.log_interaction(user_id, target_id)


async def recall_user(user_id: int):
    return await db_manager.recall_user(user_id)


async def store_memory(
    user_id: int,
    memory: str,
    topic: str = "",
    sentiment_score: float | None = None,
) -> None:
    await db_manager.store_memory(
        user_id, memory, topic=topic, sentiment_score=sentiment_score
    )


async def store_theory(subject_id: int, theory: str, confidence: float) -> None:
    return await db_manager.store_theory(subject_id, theory, confidence)


async def get_theories(subject_id: int):
    return await db_manager.get_theories(subject_id)


async def update_sentiment_trend(
    user_id: int,
    channel_id: int,
    sentiment_score: float,
) -> None:
    await db_manager.update_sentiment_trend(user_id, channel_id, sentiment_score)


async def get_sentiment_trend(user_id: int, channel_id: int):
    return await db_manager.get_sentiment_trend(user_id, channel_id)


async def get_recent_topics(limit: int = 3) -> list[str]:
    return await db_manager.get_recent_topics(limit)


async def queue_deep_reflection(user_id: int, context: dict, prompt: str) -> int:
    return await db_manager.queue_deep_reflection(user_id, context, prompt)


async def list_pending_tasks():
    return await db_manager.list_pending_tasks()


async def mark_task_done(task_id: int) -> None:
    await db_manager.mark_task_done(task_id)


async def set_do_not_mock(user_id: int, flag: bool = True) -> None:
    await db_manager.set_do_not_mock(user_id, flag)


async def is_do_not_mock(user_id: int) -> bool:
    return await db_manager.is_do_not_mock(user_id)


async def adjust_affinity(user_id: int, delta: int) -> None:
    await db_manager.adjust_affinity(user_id, delta)


async def get_affinity(user_id: int) -> int:
    return await db_manager.get_affinity(user_id)


async def set_theme(user_id: int, channel_id: int, theme: str) -> None:
    await db_manager.set_theme(user_id, channel_id, theme)


async def get_theme(user_id: int, channel_id: int):
    return await db_manager.get_theme(user_id, channel_id)


async def get_all_sentiment_trends():
    return await db_manager.get_all_sentiment_trends()


__all__ = [
    "DBManager",
    "db_manager",
    "init_db",
    "log_interaction",
    "recall_user",
    "store_memory",
    "store_theory",
    "get_theories",
    "update_sentiment_trend",
    "get_sentiment_trend",
    "get_recent_topics",
    "queue_deep_reflection",
    "list_pending_tasks",
    "mark_task_done",
    "set_do_not_mock",
    "is_do_not_mock",
    "adjust_affinity",
    "get_affinity",
    "set_theme",
    "get_theme",
    "get_all_sentiment_trends",
]
