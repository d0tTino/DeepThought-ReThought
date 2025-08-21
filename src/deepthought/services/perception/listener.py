from __future__ import annotations

import json
import logging
from typing import Any, Dict

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ...eda.events import EventSubjects
from ...eda.subscriber import Subscriber
from .service import PerceptionService

logger = logging.getLogger(__name__)


class PerceptionServiceListener:
    """Subscribe to input events and invoke :class:`PerceptionService`."""

    def __init__(
        self,
        service: PerceptionService,
        nats_client: NATS,
        js_context: JetStreamContext,
    ) -> None:
        self._service = service
        self._subscriber = Subscriber(nats_client, js_context)

    async def start(self, durable_name: str = "perception_listener") -> bool:
        """Begin listening for input events."""
        return await self._subscriber.subscribe(
            subject=EventSubjects.INPUT_RECEIVED,
            handler=self._handle,
            use_jetstream=True,
            durable=durable_name,
        )

    async def _handle(self, msg: Msg) -> None:
        """Decode event payload and dispatch to the service."""
        try:
            payload: Dict[str, Any] = json.loads(msg.data.decode())
            keys = [
                "message_id",
                "user_id",
                "spans",
                "embeddings",
                "encoders",
                "provenance",
                "text_tokens",
                "audio_path",
                "video_path",
            ]
            kwargs = {k: payload[k] for k in keys if k in payload}
            await self._service.run(**kwargs)
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to process perception input", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:  # pragma: no cover - defensive
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:  # pragma: no cover - defensive
                    logger.error("Failed to ack message after error", exc_info=True)
