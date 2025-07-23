from __future__ import annotations

import json
import logging
from typing import Optional

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext
from textblob import TextBlob

from ..eda.events import EventSubjects
from .base import BaseService
from .db_manager import DBManager
from .persona_manager import PersonaManager

logger = logging.getLogger(__name__)


class SocialGraphService(BaseService):
    """Service that records sentiment and adjusts user affinity."""

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        db_manager: Optional[DBManager] = None,
        persona_manager: Optional[PersonaManager] = None,
        *,
        nats_url: str | None = None,
        connect_retries: int = 1,
        connect_timeout: float = 2.0,
    ) -> None:
        super().__init__(
            nats_client,
            js_context,
            nats_url=nats_url,
            connect_retries=connect_retries,
            connect_timeout=connect_timeout,
        )
        self._db = db_manager or DBManager()
        self._persona = persona_manager or PersonaManager(self._db)

    async def _handle_input(self, msg: Msg) -> None:
        """Process an INPUT_RECEIVED message."""
        try:
            data = json.loads(msg.data.decode())
            text = data.get("user_input")
            if not isinstance(text, str):
                raise ValueError("user_input missing")
            score = float(TextBlob(text).sentiment.polarity)
            delta = 0
            if score > 0:
                delta = 1
            elif score < 0:
                delta = -1
            await self._db.adjust_affinity("user", delta)
            await self._persona.get_persona("user")
        except Exception:
            logger.error("Failed to handle input", exc_info=True)
        finally:
            if hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message", exc_info=True)

    async def start(self, durable_name: str = "social_graph_service") -> bool:
        self._subscriptions.clear()
        self.add_subscription(
            subject=EventSubjects.INPUT_RECEIVED,
            handler=self._handle_input,
            use_jetstream=True,
            durable=durable_name,
        )
        return await super().start()

    async def stop(self) -> None:
        if hasattr(self._db, "close"):
            try:
                await self._db.close()
            except Exception:
                logger.error("Failed to close DB", exc_info=True)
        await super().stop()
