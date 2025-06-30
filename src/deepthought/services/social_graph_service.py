import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import aiosqlite
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext


class DBManager:
    """Minimal database manager for storing social interactions."""

    def __init__(self, db_path: str = "social_graph.db") -> None:
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
            CREATE TABLE IF NOT EXISTS memories (
                user_id TEXT,
                memory TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.commit()

    async def log_interaction(
        self,
        user_id: int,
        target_id: int | None = None,
        sentiment_score: float | None = None,
    ) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            "INSERT INTO interactions (user_id, target_id) VALUES (?, ?)",
            (str(user_id), str(target_id) if target_id is not None else None),
        )
        await self._db.commit()

    async def store_memory(
        self,
        user_id: int,
        memory: str,
        topic: str = "",
        sentiment_score: float | None = None,
    ) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            "INSERT INTO memories (user_id, memory) VALUES (?, ?)",
            (str(user_id), memory),
        )
        await self._db.commit()

    async def recall_user(self, user_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT memory FROM memories WHERE user_id=?",
            (str(user_id),),
        ) as cur:
            rows = await cur.fetchall()
        return [("", r[0]) for r in rows]


from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class SocialGraphService:
    """Service that stores interactions using :class:`DBManager`."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        db: Optional[DBManager] = None,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._db = db or DBManager()

    async def _handle_input(self, msg: Msg) -> None:
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("InputReceived payload must be a dict")
            input_id = data.get("input_id")
            user_input = data.get("user_input")
            if not isinstance(input_id, str) or not isinstance(user_input, str):
                raise ValueError("Invalid input payload fields")
            logger.info("SocialGraphService received input event ID %s", input_id)

            await self._db.store_memory("user", user_input)
            await self._db.log_interaction("user", None)
            rows = await self._db.recall_user("user")
            facts = [m[1] for m in rows][-3:]
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={"facts": facts, "source": "social_graph_service"},
                input_id=input_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            await self._publisher.publish(
                EventSubjects.MEMORY_RETRIEVED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            logger.info("SocialGraphService published memory event ID %s", input_id)
            await msg.ack()
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Invalid InputReceived payload: %s", e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message after error", exc_info=True)
        except Exception as e:
            logger.error("Error in SocialGraphService handler: %s", e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message after error", exc_info=True)

    async def start(self, durable_name: str = "social_graph_listener") -> bool:
        if self._subscriber is None:
            logger.error("Subscriber not initialized for SocialGraphService.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.INPUT_RECEIVED,
                handler=self._handle_input,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("SocialGraphService subscribed to %s", EventSubjects.INPUT_RECEIVED)
            return True
        except Exception as e:
            logger.error("SocialGraphService failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("SocialGraphService stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
