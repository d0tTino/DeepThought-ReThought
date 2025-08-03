"""SQLite-based database manager for social graph interactions."""

from __future__ import annotations

import json
import os

import aiosqlite

SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "textblob").lower()
if SENTIMENT_BACKEND == "vader":
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _sentiment = SentimentIntensityAnalyzer()

        def analyze_sentiment(text: str) -> float:
            return _sentiment.polarity_scores(text)["compound"]

    except Exception:  # pragma: no cover - dependency missing
        from textblob import TextBlob

        def analyze_sentiment(text: str) -> float:
            return TextBlob(text).sentiment.polarity

else:
    from textblob import TextBlob

    def analyze_sentiment(text: str) -> float:
        return TextBlob(text).sentiment.polarity


from ..config import get_settings

# Default database path used when none is provided.
DB_PATH = get_settings().social_graph_db

# Limits used when validating input sizes
MAX_MEMORY_LENGTH = 1000
MAX_THEORY_LENGTH = 256
MAX_PROMPT_LENGTH = 2000
AFFINITY_POS_DELTA = int(os.getenv("AFFINITY_POS_DELTA", "1"))
AFFINITY_NEG_DELTA = int(os.getenv("AFFINITY_NEG_DELTA", "-1"))


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
        CREATE TABLE IF NOT EXISTS memories (
            user_id TEXT,
            topic TEXT,
            memory TEXT,
            sentiment_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
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
        CREATE TABLE IF NOT EXISTS lies (
            user_id TEXT,
            question TEXT,
            reply TEXT,
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
            if not self._initialized:
                for query in self.CREATE_TABLE_QUERIES:
                    await self._db.execute(query)
                await self._ensure_relationship_columns()
                await self._db.execute("INSERT OR IGNORE INTO trust_config (id) VALUES (1)")
                await self._db.commit()
                self._initialized = True

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _ensure_relationship_columns(self) -> None:
        """Add new columns to the relationships table if they don't exist."""
        assert self._db is not None
        async with self._db.execute("PRAGMA table_info(relationships)") as cur:
            cols = [row[1] async for row in cur]
        if "interaction_weight" not in cols:
            await self._db.execute("ALTER TABLE relationships ADD COLUMN interaction_weight REAL DEFAULT 0")
        if "last_interaction" not in cols:
            await self._db.execute("ALTER TABLE relationships ADD COLUMN last_interaction DATETIME")

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
            await self._db.execute(
                """
                INSERT INTO relationships (source_id, target_id, interaction_count, sentiment_sum, interaction_weight)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(source_id, target_id) DO UPDATE SET
                    interaction_count=relationships.interaction_count + 1,
                    sentiment_sum=relationships.sentiment_sum + excluded.sentiment_sum,
                    interaction_weight=relationships.interaction_weight + excluded.interaction_weight,
                    last_interaction=CURRENT_TIMESTAMP
                """,
                (str(user_id), str(target_id), sentiment_score or 0.0, weight),
            )
            a, b = sorted((str(user_id), str(target_id)))
            await self._db.execute(
                """
                INSERT INTO mutual_affinity (user_a, user_b, score, interaction_weight)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(user_a, user_b) DO UPDATE SET
                    score=mutual_affinity.score + 1,
                    interaction_weight=mutual_affinity.interaction_weight + excluded.interaction_weight,
                    last_interaction=CURRENT_TIMESTAMP
                """,
                (a, b, weight),
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

    async def get_theories(self, subject_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT theory, confidence FROM theories WHERE subject_id=?",
            (str(subject_id),),
        ) as cur:
            return await cur.fetchall()

    async def store_lie(self, user_id: int, question: str, reply: str) -> None:
        """Store a fabricated ``reply`` for ``question`` asked by ``user_id``."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("reply must be a non-empty string")

        await self.connect()
        assert self._db
        await self._db.execute(
            "INSERT INTO lies (user_id, question, reply) VALUES (?, ?, ?)",
            (str(user_id), question, reply),
        )
        await self._db.commit()

    async def get_last_lie(self, user_id: int, question: str) -> str | None:
        """Return the most recent fabricated reply for ``question`` by ``user_id``."""
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT reply FROM lies WHERE user_id=? AND question=? " "ORDER BY rowid DESC LIMIT 1",
            (str(user_id), question),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else None

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

    def _affinity_delta(self, value: float) -> int:
        if -1 <= float(value) <= 1:
            if value > 0:
                return AFFINITY_POS_DELTA
            if value < 0:
                return AFFINITY_NEG_DELTA
            return 0
        return int(value)

    async def adjust_affinity(self, user_id: int, delta: float) -> None:
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
        async with self._db.execute(
            "SELECT lower_limit, upper_limit, decay FROM trust_config WHERE id=1"
        ) as cur:
            row = await cur.fetchone()
            if row:
                return float(row[0]), float(row[1]), float(row[2])
            return -10.0, 10.0, 0.0

    async def set_trust_params(
        self, lower_limit: float, upper_limit: float, decay: float
    ) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            "UPDATE trust_config SET lower_limit=?, upper_limit=?, decay=? WHERE id=1",
            (float(lower_limit), float(upper_limit), float(decay)),
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
            return await cur.fetchone()

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
        a, b = sorted((str(user_a), str(user_b)))
        async with self._db.execute(
            "SELECT score FROM mutual_affinity WHERE user_a=? AND user_b=?",
            (a, b),
        ) as cur:
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0

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
