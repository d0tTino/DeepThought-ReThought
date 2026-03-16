"""SQLite-based database manager for social graph interactions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Mapping

import aiosqlite

from ..config import get_settings
from ..fact_schema import make_canonical_fact

SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "textblob").lower()
try:  # Optional dependency
    from textblob import TextBlob  # type: ignore
except Exception:  # pragma: no cover - dependency missing
    TextBlob = None  # type: ignore

if SENTIMENT_BACKEND == "vader":
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _sentiment = SentimentIntensityAnalyzer()

        def analyze_sentiment(text: str) -> float:
            return _sentiment.polarity_scores(text)["compound"]

    except Exception:  # pragma: no cover - dependency missing

        def analyze_sentiment(text: str) -> float:
            if TextBlob is None:
                return 0.0
            return TextBlob(text).sentiment.polarity
else:

    def analyze_sentiment(text: str) -> float:
        if TextBlob is None:
            return 0.0
        return TextBlob(text).sentiment.polarity


# Default database path used when none is provided.
DB_PATH = get_settings().social_graph_db

# Limits used when validating input sizes
MAX_MEMORY_LENGTH = 1000
MAX_THEORY_LENGTH = 256
MAX_PROMPT_LENGTH = 2000
AFFINITY_POS_DELTA = int(os.getenv("AFFINITY_POS_DELTA", "1"))
AFFINITY_NEG_DELTA = int(os.getenv("AFFINITY_NEG_DELTA", "-1"))


UNSET = object()


class DBManager:
    """Lightweight wrapper managing a single aiosqlite connection."""

    CREATE_TABLE_QUERIES = [
        """
        CREATE TABLE IF NOT EXISTS interactions (
            user_id TEXT,
            target_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS affinity (
            user_id TEXT PRIMARY KEY,
            score INTEGER DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS trust (
            user_id TEXT PRIMARY KEY,
            score REAL DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS trust_config (
            id INTEGER PRIMARY KEY CHECK(id=1),
            lower_limit REAL DEFAULT -10.0,
            upper_limit REAL DEFAULT 10.0,
            decay REAL DEFAULT 0.0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS trust_offenses (
            user_id TEXT PRIMARY KEY,
            manipulative_count INTEGER DEFAULT 0,
            banned_count INTEGER DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS mutual_affinity (
            user_a TEXT,
            user_b TEXT,
            score INTEGER DEFAULT 0,
            interaction_weight REAL DEFAULT 0,
            last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(user_a, user_b)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS relationships (
            source_id TEXT,
            target_id TEXT,
            interaction_count INTEGER DEFAULT 0,
            sentiment_sum REAL DEFAULT 0,
            interaction_weight REAL DEFAULT 0,
            last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(source_id, target_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS relationship_types (
            user_a TEXT,
            user_b TEXT,
            status TEXT,
            PRIMARY KEY(user_a, user_b)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS social_edges (
            source_id TEXT,
            target_id TEXT,
            edge_type TEXT,
            channel_id TEXT,
            weight REAL DEFAULT 0,
            event_count INTEGER DEFAULT 0,
            reciprocity REAL DEFAULT 0,
            sentiment_sum REAL DEFAULT 0,
            sentiment_avg REAL DEFAULT 0,
            sentiment_trend TEXT DEFAULT 'stable',
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(source_id, target_id, edge_type, channel_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS interaction_decay (
            id INTEGER PRIMARY KEY CHECK(id=1),
            weight_decay REAL DEFAULT 1.0,
            sentiment_decay REAL DEFAULT 1.0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memories (
            user_id TEXT,
            topic TEXT,
            memory TEXT,
            sentiment_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS fact_records (
            id TEXT PRIMARY KEY,
            dedup_key TEXT UNIQUE,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_value TEXT,
            object_id TEXT,
            provenance TEXT,
            confidence REAL DEFAULT 1.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            attributes TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS theories (
            subject_id TEXT,
            theory TEXT,
            confidence REAL,
            updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(subject_id, theory)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS queued_tasks (
            task_id INTEGER PRIMARY KEY,
            user_id TEXT,
            context TEXT,
            prompt TEXT,
            status TEXT DEFAULT 'pending',
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS summary_goals (
            task_id INTEGER PRIMARY KEY,
            user_id TEXT,
            context TEXT,
            prompt TEXT,
            status TEXT DEFAULT 'pending',
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_summaries (
            user_id TEXT PRIMARY KEY,
            summary TEXT,
            source_count INTEGER DEFAULT 0,
            updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sentiment_trends (
            user_id TEXT,
            channel_id TEXT,
            sentiment_sum REAL DEFAULT 0,
            message_count INTEGER DEFAULT 0,
            PRIMARY KEY(user_id, channel_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS themes (
            user_id TEXT,
            channel_id TEXT,
            theme TEXT,
            updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(user_id, channel_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_flags (
            user_id TEXT PRIMARY KEY,
            do_not_mock INTEGER
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            traits TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS recent_topics (
            topic TEXT PRIMARY KEY,
            last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS intentions (
            intention_id INTEGER PRIMARY KEY,
            goal TEXT,
            priority INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER UNIQUE,
            title TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            due_date TEXT,
            holiday INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            archived_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lies (
            quest_id INTEGER,
            question TEXT,
            reply TEXT,
            expires TIMESTAMP,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS emotions (
            user_id TEXT,
            emotion_json TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS manipulations (
            user_id TEXT,
            manipulation_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS quests (
            quest_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            quest_type TEXT,
            priority INTEGER DEFAULT 0,
            horizon TEXT,
            faction TEXT,
            cover_story TEXT,
            secrecy TEXT,
            risk TEXT,
            status TEXT DEFAULT 'pending',
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS objectives (
            objective_id INTEGER PRIMARY KEY,
            quest_id INTEGER REFERENCES quests(quest_id),
            description TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            preconditions TEXT,
            success_criteria TEXT,
            fail_criteria TEXT,
            fallbacks TEXT,
            cooldowns TEXT,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id INTEGER PRIMARY KEY,
            objective_id INTEGER REFERENCES objectives(objective_id),
            content TEXT NOT NULL,
            who TEXT,
            confidence_delta REAL DEFAULT 0,
            expiry TIMESTAMP,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS epiphanies (
            epiphany_id INTEGER PRIMARY KEY,
            quest_id INTEGER REFERENCES quests(quest_id),
            insight TEXT NOT NULL,
            who TEXT,
            confidence_delta REAL DEFAULT 0,
            expiry TIMESTAMP,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lie_ledger (
            lie_id INTEGER PRIMARY KEY,
            quest_id INTEGER REFERENCES quests(quest_id),
            lie TEXT NOT NULL,
            who TEXT,
            confidence_delta REAL DEFAULT 0,
            expiry TIMESTAMP,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._initialized = False

    async def connect(self) -> None:
        if self._db is None:
            dir_path = os.path.dirname(self.db_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row
            if not self._initialized:
                for query in self.CREATE_TABLE_QUERIES:
                    await self._db.execute(query)
                await self._ensure_relationship_columns()
                await self._ensure_social_edges_columns()
                await self._ensure_fact_records_migration()
                await self._db.execute("INSERT OR IGNORE INTO trust_config (id) VALUES (1)")
                await self._db.execute("INSERT OR IGNORE INTO interaction_decay (id) VALUES (1)")
                await self._db.commit()
                self._initialized = True

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _ensure_fact_records_migration(self) -> None:
        """Backfill canonical fact rows from legacy memories table."""
        assert self._db is not None
        async with self._db.execute("SELECT COUNT(*) FROM fact_records") as cur:
            row = await cur.fetchone()
        if row and int(row[0] or 0) > 0:
            return

        async with self._db.execute(
            "SELECT user_id, topic, memory, sentiment_score, timestamp FROM memories ORDER BY timestamp ASC"
        ) as cur:
            rows = await cur.fetchall()

        for row in rows:
            topic = str(row["topic"] or "")
            memory = str(row["memory"] or "")
            timestamp = str(row["timestamp"] or datetime.utcnow().isoformat())
            fact = make_canonical_fact(
                subject=str(row["user_id"] or "anonymous"),
                predicate="memory_note",
                object_value=memory,
                provenance={"source": "sqlite_memories", "observed_at": timestamp},
                confidence=0.6,
                created_at=timestamp,
                updated_at=timestamp,
                attributes={"topic": topic, "sentiment_score": row["sentiment_score"]},
            )
            await self._db.execute(
                """
                INSERT INTO fact_records (
                    id, dedup_key, subject, predicate, object_value, object_id,
                    provenance, confidence, created_at, updated_at, attributes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dedup_key) DO UPDATE SET
                    confidence=MAX(confidence, excluded.confidence),
                    updated_at=excluded.updated_at,
                    provenance=excluded.provenance,
                    attributes=excluded.attributes
                """,
                (
                    fact.id,
                    fact.dedup_key,
                    fact.subject,
                    fact.predicate,
                    fact.object_value,
                    fact.object_id,
                    json.dumps(fact.provenance),
                    fact.confidence,
                    fact.created_at,
                    fact.updated_at,
                    json.dumps(fact.attributes),
                ),
            )

    async def _ensure_relationship_columns(self) -> None:
        """Add new columns to the relationships table if they don't exist."""
        assert self._db is not None
        async with self._db.execute("PRAGMA table_info(relationships)") as cur:
            cols = [row[1] async for row in cur]
        if "interaction_weight" not in cols:
            await self._db.execute("ALTER TABLE relationships ADD COLUMN interaction_weight REAL DEFAULT 0")
        if "last_interaction" not in cols:
            await self._db.execute("ALTER TABLE relationships ADD COLUMN last_interaction DATETIME")

    async def _ensure_social_edges_columns(self) -> None:
        """Add new columns to the social_edges table if they don't exist."""
        assert self._db is not None
        async with self._db.execute("PRAGMA table_info(social_edges)") as cur:
            cols = [row[1] async for row in cur]
        if "channel_id" not in cols:
            await self._db.execute("ALTER TABLE social_edges ADD COLUMN channel_id TEXT")
            await self._db.execute("UPDATE social_edges SET channel_id='global' WHERE channel_id IS NULL")
        if "event_count" not in cols:
            await self._db.execute("ALTER TABLE social_edges ADD COLUMN event_count INTEGER DEFAULT 0")
            await self._db.execute("UPDATE social_edges SET event_count=CASE WHEN event_count=0 THEN 1 ELSE event_count END")
        if "reciprocity" not in cols:
            await self._db.execute("ALTER TABLE social_edges ADD COLUMN reciprocity REAL DEFAULT 0")
        if "sentiment_sum" not in cols:
            await self._db.execute("ALTER TABLE social_edges ADD COLUMN sentiment_sum REAL DEFAULT 0")
        if "sentiment_avg" not in cols:
            await self._db.execute("ALTER TABLE social_edges ADD COLUMN sentiment_avg REAL DEFAULT 0")
        if "sentiment_trend" not in cols:
            await self._db.execute("ALTER TABLE social_edges ADD COLUMN sentiment_trend TEXT DEFAULT 'stable'")

    @staticmethod
    def _normalize_timestamp_input(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(microsecond=0).isoformat()
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        raise TypeError("timestamp value must be datetime, str, or None")

    @staticmethod
    def _format_timestamp(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(microsecond=0).isoformat()
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    return parsed.replace(microsecond=0).isoformat()
                except ValueError:
                    continue
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return text
            return parsed.replace(microsecond=0).isoformat()
        return str(value)

    def _project_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "project_id": row["project_id"],
            "thread_id": row["thread_id"],
            "title": row["title"],
            "priority": row["priority"],
            "status": row["status"],
            "due_date": self._format_timestamp(row["due_date"]),
            "holiday": bool(row["holiday"]),
            "created_at": self._format_timestamp(row["created_at"]),
            "updated_at": self._format_timestamp(row["updated_at"]),
            "archived_at": self._format_timestamp(row["archived_at"]),
        }

    def _create_table_statements(self) -> list[str]:
        """Return SQL statements for creating required tables."""
        return self.CREATE_TABLE_QUERIES

    async def init_db(self) -> None:
        await self.connect()

    async def log_interaction(
        self,
        user_id: int,
        target_id: int | None = None,
        sentiment_score: float | None = None,
    ) -> None:
        await self.connect()
        assert self._db
        if sentiment_score is not None:
            if not isinstance(sentiment_score, (int, float)):
                raise ValueError("sentiment_score must be numeric")
            if not -1 <= float(sentiment_score) <= 1:
                raise ValueError("sentiment_score out of range")
        await self._db.execute(
            "INSERT INTO interactions (user_id, target_id) VALUES (?, ?)",
            (str(user_id), str(target_id) if target_id is not None else None),
        )
        delta = self._affinity_delta(sentiment_score) if sentiment_score is not None else AFFINITY_POS_DELTA
        if delta:
            await self._db.execute(
                """
                INSERT INTO affinity (user_id, score)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET score=affinity.score + ?
                """,
                (str(user_id), delta, delta),
            )
        if target_id is not None:
            weight = 1.0
            w_decay, s_decay = await self.get_decay_params()
            now = datetime.utcnow()
            # Update directional relationship with decay
            async with self._db.execute(
                "SELECT interaction_count, sentiment_sum, interaction_weight, last_interaction FROM relationships WHERE source_id=? AND target_id=?",
                (str(user_id), str(target_id)),
            ) as cur:
                row = await cur.fetchone()
            if row:
                count, ssum, w, last_ts = row
                if last_ts:
                    last_dt = datetime.fromisoformat(str(last_ts))
                    elapsed = (now - last_dt).total_seconds()
                    ssum = float(ssum) * (s_decay**elapsed)
                    w = float(w) * (w_decay**elapsed)
                count = int(count) + 1
                ssum += sentiment_score or 0.0
                w += weight
                await self._db.execute(
                    "UPDATE relationships SET interaction_count=?, sentiment_sum=?, interaction_weight=?, last_interaction=CURRENT_TIMESTAMP WHERE source_id=? AND target_id=?",
                    (count, ssum, w, str(user_id), str(target_id)),
                )
            else:
                await self._db.execute(
                    "INSERT INTO relationships (source_id, target_id, interaction_count, sentiment_sum, interaction_weight, last_interaction) VALUES (?, ?, 1, ?, ?, CURRENT_TIMESTAMP)",
                    (str(user_id), str(target_id), sentiment_score or 0.0, weight),
                )

            # Update mutual affinity with decay
            a, b = sorted((str(user_id), str(target_id)))
            async with self._db.execute(
                "SELECT score, interaction_weight, last_interaction FROM mutual_affinity WHERE user_a=? AND user_b=?",
                (a, b),
            ) as cur:
                mrow = await cur.fetchone()
            if mrow:
                score, w, last_ts = mrow
                if last_ts:
                    last_dt = datetime.fromisoformat(str(last_ts))
                    elapsed = (now - last_dt).total_seconds()
                    w = float(w) * (w_decay**elapsed)
                score = int(score) + 1
                w += weight
                await self._db.execute(
                    "UPDATE mutual_affinity SET score=?, interaction_weight=?, last_interaction=CURRENT_TIMESTAMP WHERE user_a=? AND user_b=?",
                    (score, w, a, b),
                )
            else:
                await self._db.execute(
                    "INSERT INTO mutual_affinity (user_a, user_b, score, interaction_weight, last_interaction) VALUES (?, ?, 1, ?, CURRENT_TIMESTAMP)",
                    (a, b, weight),
                )
        await self._db.commit()

    async def set_relationship(
        self,
        source_id: int,
        target_id: int,
        interaction_count: int,
        sentiment_sum: float,
        interaction_weight: float = 0.0,
        last_interaction: float | str | None = None,
    ) -> None:
        """Insert or update a relationship row with explicit values.

        This is primarily used for migrating data from legacy stores where
        interaction statistics were persisted in JSON files.
        """
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO relationships (
                source_id,
                target_id,
                interaction_count,
                sentiment_sum,
                interaction_weight,
                last_interaction
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id) DO UPDATE SET
                interaction_count=excluded.interaction_count,
                sentiment_sum=excluded.sentiment_sum,
                interaction_weight=excluded.interaction_weight,
                last_interaction=excluded.last_interaction
            """,
            (
                str(source_id),
                str(target_id),
                int(interaction_count),
                float(sentiment_sum),
                float(interaction_weight),
                last_interaction,
            ),
        )
        await self._db.commit()

    async def delete_relationship(self, source_id: int, target_id: int) -> None:
        """Remove a relationship between ``source_id`` and ``target_id``."""
        await self.connect()
        assert self._db
        await self._db.execute(
            "DELETE FROM relationships WHERE source_id=? AND target_id=?",
            (str(source_id), str(target_id)),
        )
        await self._db.commit()

    async def recall_user(self, user_id: int, limit: int | None = None):
        await self.connect()
        if limit is not None and limit <= 0:
            return []

        assert self._db
        query = (
            "SELECT json_extract(attributes, '$.topic') as topic, object_value as memory "
            "FROM fact_records WHERE subject=? ORDER BY updated_at DESC"
        )
        params: list[str | int] = [str(user_id)]
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))

        async with self._db.execute(query, params) as cur:
            rows = await cur.fetchall()
        return [(str(r[0] or ""), str(r[1] or "")) for r in rows]

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
        timestamp = datetime.utcnow().replace(microsecond=0).isoformat()
        fact = make_canonical_fact(
            subject=str(user_id),
            predicate="memory_note",
            object_value=memory.strip(),
            provenance={"source": "db_manager.store_memory", "observed_at": timestamp},
            confidence=0.7,
            created_at=timestamp,
            updated_at=timestamp,
            attributes={"topic": topic, "sentiment_score": sentiment_score},
        )
        await self._db.execute(
            """
            INSERT INTO fact_records (
                id, dedup_key, subject, predicate, object_value, object_id,
                provenance, confidence, created_at, updated_at, attributes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dedup_key) DO UPDATE SET
                confidence=MAX(fact_records.confidence, excluded.confidence),
                updated_at=excluded.updated_at,
                provenance=excluded.provenance,
                attributes=excluded.attributes
            """,
            (
                fact.id,
                fact.dedup_key,
                fact.subject,
                fact.predicate,
                fact.object_value,
                fact.object_id,
                json.dumps(fact.provenance),
                fact.confidence,
                fact.created_at,
                fact.updated_at,
                json.dumps(fact.attributes),
            ),
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

    async def store_emotion(self, user_id: int, emotions: dict | list) -> None:
        """Store a JSON-serializable emotion payload for ``user_id``."""
        if not isinstance(emotions, (dict, list)):
            raise ValueError("emotions must be a dictionary or list")
        try:
            emotion_json = json.dumps(emotions)
        except (TypeError, ValueError) as exc:  # pragma: no cover - json failure
            raise ValueError("emotions must be JSON serializable") from exc

        await self.connect()
        assert self._db
        await self._db.execute(
            "INSERT INTO emotions (user_id, emotion_json) VALUES (?, ?)",
            (str(user_id), emotion_json),
        )
        await self._db.commit()

    async def store_theory(self, subject_id: int, theory: str, confidence: float) -> None:
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


    async def adjust_theory_confidence(self, subject_id: int | str, delta: float) -> None:
        """Adjust confidence for all theories associated with ``subject_id``."""
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            UPDATE theories
            SET confidence = MIN(1.0, MAX(0.0, confidence + ?)),
                updated=CURRENT_TIMESTAMP
            WHERE subject_id=?
            """,
            (float(delta), str(subject_id)),
        )
        await self._db.commit()

    async def upsert_user_summary(self, user_id: int | str, summary: str, source_count: int) -> None:
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summary must be a non-empty string")
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO user_summaries (user_id, summary, source_count, updated)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                summary=excluded.summary,
                source_count=excluded.source_count,
                updated=CURRENT_TIMESTAMP
            """,
            (str(user_id), summary.strip(), int(source_count)),
        )
        await self._db.commit()

    async def get_user_summary(self, user_id: int | str) -> tuple[str, int, str] | None:
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT summary, source_count, updated FROM user_summaries WHERE user_id=?",
            (str(user_id),),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return str(row[0]), int(row[1] or 0), str(row[2])

    async def list_users_with_long_history(self, min_memories: int = 25) -> list[tuple[str, int]]:
        await self.connect()
        assert self._db
        async with self._db.execute(
            """
            SELECT user_id, COUNT(*) as memory_count
            FROM memories
            GROUP BY user_id
            HAVING COUNT(*) >= ?
            ORDER BY COUNT(*) DESC
            """,
            (int(min_memories),),
        ) as cur:
            rows = await cur.fetchall()
        return [(str(user_id), int(count)) for user_id, count in rows]

    async def get_theories(self, subject_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT theory, confidence FROM theories WHERE subject_id=?",
            (str(subject_id),),
        ) as cur:
            return await cur.fetchall()

    async def store_lie(self, quest_id: int, question: str, reply: str, ttl: int = 3600) -> None:
        """Store a fabricated ``reply`` for ``question`` tied to ``quest_id``.

        ``ttl`` is the number of seconds before the record expires.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("reply must be a non-empty string")

        await self.connect()
        assert self._db
        expires = datetime.utcnow() + timedelta(seconds=int(ttl))
        expires_str = expires.strftime("%Y-%m-%d %H:%M:%S")
        await self._db.execute(
            "INSERT INTO lies (quest_id, question, reply, expires) VALUES (?, ?, ?, ?)",
            (quest_id, question, reply, expires_str),
        )
        await self._db.commit()

    async def get_last_lie(self, quest_id: int, question: str) -> str | None:
        """Return the most recent lie for ``question`` by ``quest_id`` if not expired."""
        await self.connect()
        assert self._db
        await self._db.execute("DELETE FROM lies WHERE expires <= CURRENT_TIMESTAMP")
        await self._db.commit()
        async with self._db.execute(
            "SELECT reply FROM lies WHERE quest_id=? AND question=? AND expires > CURRENT_TIMESTAMP "
            "ORDER BY rowid DESC LIMIT 1",
            (quest_id, question),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else None

    async def list_lies(self, limit: int = 20) -> list[tuple[int, int, str, str]]:
        """Return recent non-expired entries from the ``lies`` table.

        Each result is ``(rowid, quest_id, question, reply)`` ordered by
        ``rowid`` descending.
        """
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT rowid, quest_id, question, reply FROM lies "
            "WHERE expires > CURRENT_TIMESTAMP ORDER BY rowid DESC LIMIT ?",
            (limit,),
        ) as cur:
            return await cur.fetchall()

    async def delete_lie(self, rowid: int) -> bool:
        """Delete a lie by its ``rowid``.

        Returns ``True`` if a row was removed.
        """
        await self.connect()
        assert self._db
        cur = await self._db.execute("DELETE FROM lies WHERE rowid=?", (rowid,))
        await self._db.commit()
        return cur.rowcount > 0

    async def update_lie(self, rowid: int, reply: str) -> bool:
        """Update the ``reply`` field of a lie specified by ``rowid``."""
        if not reply.strip():
            raise ValueError("reply must be a non-empty string")

        await self.connect()
        assert self._db
        cur = await self._db.execute(
            "UPDATE lies SET reply=? WHERE rowid=?",
            (reply, rowid),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def log_manipulation(self, user_id: int, category: str) -> None:
        """Record a manipulation ``category`` associated with ``user_id``."""
        if not isinstance(category, str) or not category.strip():
            raise ValueError("category must be a non-empty string")

        await self.connect()
        assert self._db
        await self._db.execute(
            "INSERT INTO manipulations (user_id, manipulation_type) VALUES (?, ?)",
            (str(user_id), category),
        )
        await self._db.commit()

    async def record_emotion(self, user_id: int, emotions: dict | list) -> None:
        """Alias for :meth:`store_emotion` for backward compatibility."""
        await self.store_emotion(user_id, emotions)

    async def record_manipulation(self, user_id: int, category: str) -> None:
        """Alias for :meth:`log_manipulation` for backward compatibility."""
        await self.log_manipulation(user_id, category)

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

    async def update_relationship_trend(
        self,
        source_id: int,
        target_id: int,
        sentiment_score: float,
    ) -> None:
        """Update running sentiment stats for a user pair.

        This stores a rolling average by tracking cumulative sentiment sum
        and interaction count for ``source_id`` -> ``target_id``.
        """
        if not isinstance(sentiment_score, (int, float)):
            raise ValueError("sentiment_score must be numeric")
        if not -1 <= float(sentiment_score) <= 1:
            raise ValueError("sentiment_score out of range")
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO relationships (source_id, target_id, interaction_count, sentiment_sum, last_interaction)
            VALUES (?, ?, 1, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source_id, target_id) DO UPDATE SET
                interaction_count=relationships.interaction_count + 1,
                sentiment_sum=relationships.sentiment_sum + excluded.sentiment_sum,
                last_interaction=CURRENT_TIMESTAMP
            """,
            (str(source_id), str(target_id), sentiment_score),
        )
        await self._db.commit()

    async def get_relationship_trend(self, source_id: int, target_id: int):
        """Return the average sentiment and count for a user pair."""
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT sentiment_sum, interaction_count FROM relationships WHERE source_id=? AND target_id=?",
            (str(source_id), str(target_id)),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        sentiment_sum, count = row
        avg = float(sentiment_sum) / count if count else 0.0
        return avg, int(count)

    async def queue_deep_reflection(self, user_id: int, context: dict, prompt: str) -> int:
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

    async def add_summary_goal(self, user_id: int, context: dict, prompt: str) -> int:
        """Store a generated summary and goal using the queued task schema."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(context, dict):
            raise ValueError("context must be a dictionary")
        try:
            context_json = json.dumps(context)
        except (TypeError, ValueError) as exc:
            raise ValueError("context is not JSON serializable") from exc

        await self.connect()
        assert self._db
        cur = await self._db.execute(
            "INSERT INTO summary_goals (user_id, context, prompt) VALUES (?, ?, ?)",
            (str(user_id), context_json, prompt),
        )
        await self._db.commit()
        return cur.lastrowid

    async def list_pending_summary_goals(self):
        """Return pending summary/goal tasks."""
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT task_id, user_id, context, prompt FROM summary_goals WHERE status='pending'"
        ) as cur:
            return await cur.fetchall()

    async def mark_summary_goal_done(self, task_id: int) -> None:
        """Mark a stored summary goal as completed."""
        await self.connect()
        assert self._db
        await self._db.execute(
            "UPDATE summary_goals SET status='done' WHERE task_id=?",
            (task_id,),
        )
        await self._db.commit()

    async def create_project(
        self,
        thread_id: int | None,
        title: str,
        *,
        priority: int = 0,
        status: str = "active",
        due_date: datetime | str | None = None,
        holiday: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status must be a non-empty string")
        await self.connect()
        assert self._db
        due_value = self._normalize_timestamp_input(due_date)
        cur = await self._db.execute(
            """
            INSERT INTO projects (thread_id, title, priority, status, due_date, holiday)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(thread_id) if thread_id is not None else None,
                title.strip(),
                int(priority),
                status.strip(),
                due_value,
                int(bool(holiday)),
            ),
        )
        await self._db.commit()
        project_id = cur.lastrowid
        if project_id is None:
            raise RuntimeError("Failed to determine project identifier")
        project = await self.get_project(project_id)
        if project is None:
            raise RuntimeError("Failed to load project after creation")
        return project

    async def update_project(
        self,
        thread_id: int,
        *,
        title: str | None = None,
        priority: int | None = None,
        status: str | None = None,
        due_date: datetime | str | None | object = UNSET,
        holiday: bool | None = None,
    ) -> dict[str, Any] | None:
        await self.connect()
        assert self._db
        clauses: list[str] = []
        params: list[Any] = []
        if title is not None:
            if not isinstance(title, str) or not title.strip():
                raise ValueError("title must be a non-empty string")
            clauses.append("title = ?")
            params.append(title.strip())
        if priority is not None:
            clauses.append("priority = ?")
            params.append(int(priority))
        if status is not None:
            if not isinstance(status, str) or not status.strip():
                raise ValueError("status must be a non-empty string")
            clauses.append("status = ?")
            params.append(status.strip())
        if due_date is not UNSET:
            due_value = self._normalize_timestamp_input(due_date)
            clauses.append("due_date = ?")
            params.append(due_value)
        if holiday is not None:
            clauses.append("holiday = ?")
            params.append(int(bool(holiday)))
        if not clauses:
            return await self.get_project_by_thread(thread_id)
        clauses.append("updated_at = CURRENT_TIMESTAMP")
        query = f"UPDATE projects SET {', '.join(clauses)} WHERE thread_id = ?"
        params.append(int(thread_id))
        cur = await self._db.execute(query, params)
        await self._db.commit()
        if cur.rowcount == 0:
            return None
        return await self.get_project_by_thread(thread_id)

    async def get_project(self, project_id: int) -> dict[str, Any] | None:
        await self.connect()
        assert self._db
        async with self._db.execute(
            """
            SELECT project_id, thread_id, title, priority, status, due_date, holiday, created_at, updated_at, archived_at
            FROM projects
            WHERE project_id = ?
            """,
            (int(project_id),),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return self._project_row_to_dict(row)

    async def get_project_by_thread(self, thread_id: int | None) -> dict[str, Any] | None:
        await self.connect()
        assert self._db
        if thread_id is None:
            query = (
                "SELECT project_id, thread_id, title, priority, status, due_date, holiday, created_at, updated_at, archived_at "
                "FROM projects WHERE thread_id IS NULL"
            )
            params = ()
        else:
            query = (
                "SELECT project_id, thread_id, title, priority, status, due_date, holiday, created_at, updated_at, archived_at "
                "FROM projects WHERE thread_id = ?"
            )
            params = (int(thread_id),)
        async with self._db.execute(query, params) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return self._project_row_to_dict(row)

    async def list_projects(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        await self.connect()
        assert self._db
        query = (
            "SELECT project_id, thread_id, title, priority, status, due_date, holiday, created_at, updated_at, archived_at "
            "FROM projects"
        )
        if not include_archived:
            query += " WHERE archived_at IS NULL"
        query += " ORDER BY priority DESC, COALESCE(due_date, ''), title"
        async with self._db.execute(query) as cur:
            rows = await cur.fetchall()
        return [self._project_row_to_dict(row) for row in rows]

    async def archive_project(
        self,
        thread_id: int,
        *,
        status: str | None = "archived",
    ) -> dict[str, Any] | None:
        await self.connect()
        assert self._db
        clauses = ["archived_at = CURRENT_TIMESTAMP", "updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = []
        if status is not None:
            if not isinstance(status, str) or not status.strip():
                raise ValueError("status must be a non-empty string")
            clauses.append("status = ?")
            params.append(status.strip())
        query = f"UPDATE projects SET {', '.join(clauses)} WHERE thread_id = ?"
        params.append(int(thread_id))
        cur = await self._db.execute(query, params)
        await self._db.commit()
        if cur.rowcount == 0:
            return None
        return await self.get_project_by_thread(thread_id)

    async def add_intention(self, goal: str, priority: int) -> int:
        """Queue an intention with ``goal`` and ``priority``."""
        await self.connect()
        assert self._db
        cur = await self._db.execute(
            "INSERT INTO intentions (goal, priority) VALUES (?, ?)",
            (goal, int(priority)),
        )
        await self._db.commit()
        return cur.lastrowid

    async def list_pending_intentions(self):
        """Return pending intentions sorted by priority."""
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT intention_id, goal, priority FROM intentions "
            "WHERE status='pending' ORDER BY priority DESC, intention_id"
        ) as cur:
            return await cur.fetchall()

    async def mark_intention_done(self, intention_id: int) -> None:
        """Mark an intention as completed."""
        await self.connect()
        assert self._db
        await self._db.execute(
            "UPDATE intentions SET status='done' WHERE intention_id=?",
            (intention_id,),
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

    async def set_user_profile(self, user_id: int, traits: dict | list | str) -> None:
        await self.connect()
        assert self._db
        data = json.dumps(traits) if not isinstance(traits, str) else traits
        await self._db.execute(
            """
            INSERT INTO user_profiles (user_id, traits)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET traits=excluded.traits
            """,
            (str(user_id), data),
        )
        await self._db.commit()

    async def get_user_profile(self, user_id: int) -> dict | list | str | None:
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT traits FROM user_profiles WHERE user_id=?",
            (str(user_id),),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return row[0]

    def _affinity_delta(self, value: float) -> int:
        if -1 <= float(value) <= 1:
            if value > 0:
                return AFFINITY_POS_DELTA
            if value < 0:
                return AFFINITY_NEG_DELTA
            return 0
        return int(round(float(value)))

    @staticmethod
    def sentiment_to_affinity_delta(
        sentiment_score: float,
        *,
        scale: float = 3.0,
        cap: int = 3,
    ) -> int:
        """Map sentiment polarity in ``[-1, 1]`` to a small integer affinity step."""
        if not isinstance(sentiment_score, (int, float)):
            raise ValueError("sentiment_score must be numeric")
        if not -1 <= float(sentiment_score) <= 1:
            raise ValueError("sentiment_score out of range")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if cap < 0:
            raise ValueError("cap must be non-negative")

        delta = int(round(float(sentiment_score) * scale))
        if cap:
            delta = max(-cap, min(cap, delta))
        return delta

    async def adjust_affinity(
        self,
        user_id: int,
        delta: float,
        target_id: int | None = None,
    ) -> None:
        """Adjust affinity for a user and optionally update pairwise state.

        When ``target_id`` is provided, this mirrors ``log_interaction`` by
        updating ``relationships`` (directional) and ``mutual_affinity``
        (symmetric) rows alongside the global ``affinity`` score.
        """
        delta_int = self._affinity_delta(delta)
        if not delta_int:
            return
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO affinity (user_id, score)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET score=affinity.score + ?
            """,
            (str(user_id), delta_int, delta_int),
        )
        if target_id is not None:
            weight = 1.0
            w_decay, s_decay = await self.get_decay_params()
            now = datetime.utcnow()
            async with self._db.execute(
                "SELECT interaction_count, sentiment_sum, interaction_weight, last_interaction FROM relationships WHERE source_id=? AND target_id=?",
                (str(user_id), str(target_id)),
            ) as cur:
                row = await cur.fetchone()
            if row:
                count, ssum, w, last_ts = row
                if last_ts:
                    last_dt = datetime.fromisoformat(str(last_ts))
                    elapsed = (now - last_dt).total_seconds()
                    ssum = float(ssum) * (s_decay**elapsed)
                    w = float(w) * (w_decay**elapsed)
                count = int(count) + 1
                ssum += float(delta)
                w += weight
                await self._db.execute(
                    "UPDATE relationships SET interaction_count=?, sentiment_sum=?, interaction_weight=?, last_interaction=CURRENT_TIMESTAMP WHERE source_id=? AND target_id=?",
                    (count, ssum, w, str(user_id), str(target_id)),
                )
            else:
                await self._db.execute(
                    "INSERT INTO relationships (source_id, target_id, interaction_count, sentiment_sum, interaction_weight, last_interaction) VALUES (?, ?, 1, ?, ?, CURRENT_TIMESTAMP)",
                    (str(user_id), str(target_id), float(delta), weight),
                )

            a, b = sorted((str(user_id), str(target_id)))
            async with self._db.execute(
                "SELECT score, interaction_weight, last_interaction FROM mutual_affinity WHERE user_a=? AND user_b=?",
                (a, b),
            ) as cur:
                mrow = await cur.fetchone()
            if mrow:
                score, w, last_ts = mrow
                if last_ts:
                    last_dt = datetime.fromisoformat(str(last_ts))
                    elapsed = (now - last_dt).total_seconds()
                    w = float(w) * (w_decay**elapsed)
                score = int(score) + delta_int
                w += weight
                await self._db.execute(
                    "UPDATE mutual_affinity SET score=?, interaction_weight=?, last_interaction=CURRENT_TIMESTAMP WHERE user_a=? AND user_b=?",
                    (score, w, a, b),
                )
            else:
                await self._db.execute(
                    "INSERT INTO mutual_affinity (user_a, user_b, score, interaction_weight, last_interaction) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (a, b, delta_int, weight),
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

    async def adjust_trust(self, user_id: int, delta: float) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO trust (user_id, score)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET score=trust.score + ?
            """,
            (str(user_id), float(delta), float(delta)),
        )
        await self._db.commit()

    async def get_trust(self, user_id: int) -> float:
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT score FROM trust WHERE user_id=?",
            (str(user_id),),
        ) as cur:
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0

    async def get_trust_params(self) -> tuple[float, float, float]:
        await self.connect()
        assert self._db
        async with self._db.execute("SELECT lower_limit, upper_limit, decay FROM trust_config WHERE id=1") as cur:
            row = await cur.fetchone()
            if row:
                return float(row[0]), float(row[1]), float(row[2])
            return -10.0, 10.0, 0.0

    async def set_trust_params(self, lower_limit: float, upper_limit: float, decay: float) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            "UPDATE trust_config SET lower_limit=?, upper_limit=?, decay=? WHERE id=1",
            (float(lower_limit), float(upper_limit), float(decay)),
        )
        await self._db.commit()

    async def get_decay_params(self) -> tuple[float, float]:
        """Return stored decay factors for weights and sentiment."""
        await self.connect()
        assert self._db
        async with self._db.execute("SELECT weight_decay, sentiment_decay FROM interaction_decay WHERE id=1") as cur:
            row = await cur.fetchone()
            if row:
                return float(row[0]), float(row[1])
            return 1.0, 1.0

    async def set_decay_params(self, weight_decay: float, sentiment_decay: float) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            "UPDATE interaction_decay SET weight_decay=?, sentiment_decay=? WHERE id=1",
            (float(weight_decay), float(sentiment_decay)),
        )
        await self._db.commit()

    async def increment_offense(self, user_id: int, offense: str) -> int:
        column = "manipulative_count" if offense == "manipulative" else "banned_count"
        await self.connect()
        assert self._db
        await self._db.execute(
            f"""
            INSERT INTO trust_offenses (user_id, {column})
            VALUES (?, 1)
            ON CONFLICT(user_id) DO UPDATE SET {column} = {column} + 1
            """,
            (str(user_id),),
        )
        await self._db.commit()
        async with self._db.execute(
            f"SELECT {column} FROM trust_offenses WHERE user_id=?",
            (str(user_id),),
        ) as cur:
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def get_mutual_affinity(self, user_id: int) -> float:
        """Return a combined score from affinity and trust for ``user_id``."""
        affinity = await self.get_affinity(user_id)
        trust = await self.get_trust(user_id)
        return float(affinity) + float(trust)

    async def get_relationship(self, user_id: int, target_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT interaction_count, sentiment_sum, interaction_weight, last_interaction FROM relationships WHERE source_id=? AND target_id=?",
            (str(user_id), str(target_id)),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        count, sentiment_sum, weight, last_ts = row
        w_decay, s_decay = await self.get_decay_params()
        if last_ts:
            last_dt = datetime.fromisoformat(str(last_ts))
            elapsed = (datetime.utcnow() - last_dt).total_seconds()
            sentiment_sum = float(sentiment_sum) * (s_decay**elapsed)
            weight = float(weight) * (w_decay**elapsed)
        return count, sentiment_sum, weight, last_ts

    async def _get_relationship_avg(self, user_id: int, target_id: int) -> float:
        row = await self.get_relationship(user_id, target_id)
        if not row or not row[0]:
            return 0.0
        count, sentiment_sum = row[0], row[1]
        return float(sentiment_sum) / count

    async def get_friendliness(self, user_id: int, target_id: int) -> float:
        avg = await self._get_relationship_avg(user_id, target_id)
        return max(0.0, avg)

    async def get_hostility(self, user_id: int, target_id: int) -> float:
        avg = await self._get_relationship_avg(user_id, target_id)
        return min(0.0, avg)

    async def get_interaction_weight(self, user_id: int, target_id: int) -> float:
        row = await self.get_relationship(user_id, target_id)
        return float(row[2]) if row and row[2] is not None else 0.0

    async def get_last_interaction(self, user_id: int, target_id: int):
        row = await self.get_relationship(user_id, target_id)
        return row[3] if row else None

    async def get_pair_mutual_affinity(self, user_a: int, user_b: int) -> float:
        await self.connect()
        assert self._db
        w_decay, _ = await self.get_decay_params()
        a, b = sorted((str(user_a), str(user_b)))
        async with self._db.execute(
            "SELECT interaction_weight, last_interaction FROM mutual_affinity WHERE user_a=? AND user_b=?",
            (a, b),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return 0.0
        weight, last_ts = row
        if last_ts:
            last_dt = datetime.fromisoformat(str(last_ts))
            elapsed = (datetime.utcnow() - last_dt).total_seconds()
            weight = float(weight) * (w_decay**elapsed)
        return float(weight)

    async def set_relationship_type(self, user_a: int, user_b: int, status: str) -> None:
        await self.connect()
        assert self._db
        a, b = sorted((str(user_a), str(user_b)))
        await self._db.execute(
            """
            INSERT INTO relationship_types (user_a, user_b, status)
            VALUES (?, ?, ?)
            ON CONFLICT(user_a, user_b) DO UPDATE SET status=excluded.status
            """,
            (a, b, status),
        )
        await self._db.commit()

    async def get_relationship_type(self, user_a: int, user_b: int) -> str | None:
        await self.connect()
        assert self._db
        a, b = sorted((str(user_a), str(user_b)))
        async with self._db.execute(
            "SELECT status FROM relationship_types WHERE user_a=? AND user_b=?",
            (a, b),
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    async def update_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: str,
        weight_delta: float,
        *,
        channel_id: str | None = None,
        sentiment_score: float | None = None,
        event_count_delta: int = 1,
    ) -> None:
        """Insert or update a typed edge applying decay to the stored weight."""
        await self.connect()
        assert self._db
        w_decay, _ = await self.get_decay_params()
        now = datetime.utcnow()
        channel_key = str(channel_id) if channel_id else "global"
        async with self._db.execute(
            "SELECT weight, event_count, sentiment_sum, last_updated FROM social_edges WHERE source_id=? AND target_id=? AND edge_type=? AND channel_id=?",
            (str(source_id), str(target_id), edge_type, channel_key),
        ) as cur:
            row = await cur.fetchone()

        sentiment_value = float(sentiment_score) if sentiment_score is not None else 0.0
        event_inc = max(0, int(event_count_delta))
        if row:
            weight, event_count, sentiment_sum, last_ts = row
            if last_ts:
                last_dt = datetime.fromisoformat(str(last_ts))
                elapsed = (now - last_dt).total_seconds()
                weight = float(weight) * (w_decay**elapsed)
            updated_weight = float(weight) + float(weight_delta)
            updated_events = int(event_count or 0) + event_inc
            updated_sentiment_sum = float(sentiment_sum or 0.0) + sentiment_value
            updated_sentiment_avg = (
                updated_sentiment_sum / updated_events if updated_events else 0.0
            )
            trend = "up" if updated_sentiment_avg > 0.2 else "down" if updated_sentiment_avg < -0.2 else "stable"
            await self._db.execute(
                """
                UPDATE social_edges
                SET weight=?,
                    event_count=?,
                    sentiment_sum=?,
                    sentiment_avg=?,
                    sentiment_trend=?,
                    last_updated=CURRENT_TIMESTAMP
                WHERE source_id=? AND target_id=? AND edge_type=? AND channel_id=?
                """,
                (
                    updated_weight,
                    updated_events,
                    updated_sentiment_sum,
                    updated_sentiment_avg,
                    trend,
                    str(source_id),
                    str(target_id),
                    edge_type,
                    channel_key,
                ),
            )
        else:
            initial_events = event_inc or 1
            initial_avg = sentiment_value / initial_events
            trend = "up" if initial_avg > 0.2 else "down" if initial_avg < -0.2 else "stable"
            await self._db.execute(
                """
                INSERT INTO social_edges (
                    source_id, target_id, edge_type, channel_id, weight,
                    event_count, reciprocity, sentiment_sum, sentiment_avg,
                    sentiment_trend, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    str(source_id),
                    str(target_id),
                    edge_type,
                    channel_key,
                    float(weight_delta),
                    initial_events,
                    sentiment_value,
                    initial_avg,
                    trend,
                ),
            )

        await self._recompute_reciprocity(source_id, target_id, edge_type, channel_key)
        await self._db.commit()

    async def _recompute_reciprocity(self, source_id: int, target_id: int, edge_type: str, channel_key: str) -> None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT weight FROM social_edges WHERE source_id=? AND target_id=? AND edge_type=? AND channel_id=?",
            (str(source_id), str(target_id), edge_type, channel_key),
        ) as cur:
            forward = await cur.fetchone()
        async with self._db.execute(
            "SELECT weight FROM social_edges WHERE source_id=? AND target_id=? AND edge_type=? AND channel_id=?",
            (str(target_id), str(source_id), edge_type, channel_key),
        ) as cur:
            backward = await cur.fetchone()
        fw = abs(float(forward[0])) if forward else 0.0
        bw = abs(float(backward[0])) if backward else 0.0
        reciprocity = (2 * min(fw, bw) / (fw + bw)) if (fw + bw) > 0 else 0.0
        await self._db.execute(
            "UPDATE social_edges SET reciprocity=? WHERE source_id=? AND target_id=? AND edge_type=? AND channel_id=?",
            (reciprocity, str(source_id), str(target_id), edge_type, channel_key),
        )
        if backward:
            await self._db.execute(
                "UPDATE social_edges SET reciprocity=? WHERE source_id=? AND target_id=? AND edge_type=? AND channel_id=?",
                (reciprocity, str(target_id), str(source_id), edge_type, channel_key),
            )

    async def get_edge_weight(
        self,
        source_id: int,
        target_id: int,
        edge_type: str,
        *,
        channel_id: str | None = None,
    ) -> float:
        """Return the decayed weight for the edge between ``source_id`` and ``target_id``."""
        await self.connect()
        assert self._db
        w_decay, _ = await self.get_decay_params()
        channel_key = str(channel_id) if channel_id else "global"
        async with self._db.execute(
            "SELECT weight, last_updated FROM social_edges WHERE source_id=? AND target_id=? AND edge_type=? AND channel_id=?",
            (str(source_id), str(target_id), edge_type, channel_key),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return 0.0
        weight, last_ts = row
        if last_ts:
            last_dt = datetime.fromisoformat(str(last_ts))
            elapsed = (datetime.utcnow() - last_dt).total_seconds()
            weight = float(weight) * (w_decay**elapsed)
        return float(weight)

    async def get_edges(
        self,
        edge_type: str | None = None,
        *,
        channel_id: str | None = None,
    ) -> list[tuple[str, str, float]]:
        """Return all edges, optionally filtered by ``edge_type`` with decay applied."""
        await self.connect()
        assert self._db
        w_decay, _ = await self.get_decay_params()
        query = "SELECT source_id, target_id, edge_type, weight, last_updated FROM social_edges"
        clauses: list[str] = []
        params_list: list[str] = []
        if edge_type:
            clauses.append("edge_type=?")
            params_list.append(edge_type)
        if channel_id:
            clauses.append("channel_id=?")
            params_list.append(str(channel_id))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        params = tuple(params_list)
        results: list[tuple[str, str, float]] = []
        async with self._db.execute(query, params) as cur:
            async for src, tgt, etype, weight, last_ts in cur:
                if last_ts:
                    last_dt = datetime.fromisoformat(str(last_ts))
                    elapsed = (datetime.utcnow() - last_dt).total_seconds()
                    weight = float(weight) * (w_decay**elapsed)
                results.append((src, tgt, float(weight)))
        return results

    async def get_edge_summary(
        self,
        source_id: int,
        target_id: int,
        edge_type: str,
        *,
        channel_id: str | None = None,
    ) -> dict[str, Any]:
        await self.connect()
        assert self._db
        channel_key = str(channel_id) if channel_id else "global"
        async with self._db.execute(
            """
            SELECT weight, event_count, reciprocity, sentiment_avg, sentiment_trend, last_updated
            FROM social_edges
            WHERE source_id=? AND target_id=? AND edge_type=? AND channel_id=?
            """,
            (str(source_id), str(target_id), edge_type, channel_key),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return {
                "weight": 0.0,
                "event_count": 0,
                "reciprocity": 0.0,
                "sentiment_avg": 0.0,
                "sentiment_trend": "stable",
                "last_updated": None,
            }
        return {
            "weight": float(row[0] or 0.0),
            "event_count": int(row[1] or 0),
            "reciprocity": float(row[2] or 0.0),
            "sentiment_avg": float(row[3] or 0.0),
            "sentiment_trend": str(row[4] or "stable"),
            "last_updated": self._format_timestamp(row[5]),
        }

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

    async def get_recent_topics(self, limit: int = 3) -> list[str]:
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT topic FROM recent_topics ORDER BY last_used DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
            return [r[0] for r in rows]
