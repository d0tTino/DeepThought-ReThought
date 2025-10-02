"""Shared helpers for DeepThought asynchronous services."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from ..bus import Publisher, Subscriber

logger = logging.getLogger(__name__)


class BaseService:
    """Base class that provides convenience helpers for services."""

    def __init__(self, subscriber: Subscriber, publisher: Publisher) -> None:
        self._subscriber = subscriber
        self._publisher = publisher

    @staticmethod
    def _decode_message(msg: Any) -> Dict[str, Any]:
        """Decode a NATS message into a dictionary."""
        try:
            if not hasattr(msg, "data"):
                raise ValueError("Message does not contain raw data")
            payload = msg.data
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            if isinstance(payload, str):
                return json.loads(payload)
            if isinstance(payload, Dict):
                return payload
            raise TypeError(f"Unsupported payload type: {type(payload)!r}")
        except Exception:
            logger.exception("Failed to decode message payload.")
            raise

    @staticmethod
    async def _ack_message(msg: Any) -> None:
        """Safely acknowledge a JetStream message if possible."""
        if hasattr(msg, "ack") and callable(msg.ack):
            try:
                await msg.ack()
            except Exception:
                logger.exception("Failed to acknowledge JetStream message.")
                raise

    async def _publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """Publish a payload and log failures."""
        try:
            await self._publisher.publish(subject, payload)
        except Exception:
            logger.exception("Failed to publish payload to %s", subject)
            raise
