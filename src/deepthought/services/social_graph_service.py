from __future__ import annotations

import logging
from typing import Optional

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext
from .cognitive_core_service import CognitiveCoreService

from ..eda.events import EventSubjects
from .base import BaseService
from .db_manager import DBManager
from .persona_manager import PersonaManager
from .prism_adapter import PrismAdapter
from .social_graph_memory import SocialGraphMemory

logger = logging.getLogger(__name__)


class SocialGraphService(BaseService):
    """Service that records sentiment and adjusts user affinity."""

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        db_manager: Optional[DBManager] = None,
        persona_manager: Optional[PersonaManager] = None,
        cognitive_core: CognitiveCoreService | None = None,
        prism_adapter: Optional[PrismAdapter] = None,
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
        self._memory = SocialGraphMemory(self._db)
        self._persona = persona_manager or PersonaManager(self._db)
        if cognitive_core is None:
            raise ValueError("cognitive_core service is required")
        self._core = cognitive_core
        self._prism = prism_adapter or PrismAdapter(self._memory)

    async def _handle_input(self, msg: Msg) -> None:
        """Process an INPUT_RECEIVED message."""
        try:
            await self._core._handle_input(msg)
            await self._persona.get_persona("user")
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to handle input", exc_info=True)

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
