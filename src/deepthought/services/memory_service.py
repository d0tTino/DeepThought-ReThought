import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..config import Settings, get_settings
from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..memory import create_memory_backend
from ..memory.tiered import TieredMemory
from ..metrics.prometheus import INPUT_LATENCY_SECONDS, INPUTS_TOTAL

logger = logging.getLogger(__name__)


from .base import BaseService


class MemoryService(BaseService):
    """Service that stores and retrieves interactions using :class:`TieredMemory`."""

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        settings: Settings | None = None,
        memory: Optional[TieredMemory] = None,
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
        settings = settings or get_settings()
        if memory is None:
            memory = create_memory_backend(settings=settings)
        self._memory = memory

    @classmethod
    def from_config(
        cls,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        *,
        nats_url: str | None = None,
        connect_retries: int = 1,
        connect_timeout: float = 2.0,
    ) -> "MemoryService":
        """Return a service with backends configured from environment variables."""

        settings = get_settings()
        return cls(
            nats_client,
            js_context,
            settings,
            nats_url=nats_url,
            connect_retries=connect_retries,
            connect_timeout=connect_timeout,
        )

    async def _handle_input(self, msg: Msg) -> None:
        input_id = "unknown"
        start = time.perf_counter()
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("InputReceived payload must be a dict")
            input_id = data.get("input_id")
            user_input = data.get("user_input")
            if not isinstance(input_id, str) or not isinstance(user_input, str):
                raise ValueError("Invalid input payload fields")
            logger.info("MemoryService received input event ID %s", input_id)

            self._memory.store_interaction(user_input)
            facts = self._memory.retrieve_context(user_input)
            memory_data = {"facts": facts, "source": "memory_service"}
            payload = MemoryRetrievedPayload(
                retrieved_knowledge=memory_data,
                input_id=input_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            await self._publisher.publish(EventSubjects.MEMORY_RETRIEVED, payload, use_jetstream=True, timeout=10.0)
            logger.info("MemoryService published memory event ID %s", input_id)
            if hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message", exc_info=True)
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
            logger.error("Error in MemoryService handler: %s", e, exc_info=True)
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
        finally:
            duration = time.perf_counter() - start
            INPUTS_TOTAL.labels(service="memory_service").inc()
            INPUT_LATENCY_SECONDS.labels(service="memory_service").observe(duration)

    async def start(self, durable_name: str = "memory_service_listener") -> bool:
        self._subscriptions.clear()
        self.add_subscription(
            subject=EventSubjects.INPUT_RECEIVED,
            handler=self._handle_input,
            use_jetstream=True,
            durable=durable_name,
        )
        started = await super().start()
        if started:
            logger.info("MemoryService subscribed to %s", EventSubjects.INPUT_RECEIVED)
        return started

    async def stop(self) -> None:
        try:
            backend = getattr(self._memory, "graph_backend", None)
            connector = None
            if hasattr(backend, "_dal"):
                connector = getattr(getattr(backend, "_dal", None), "_connector", None)
            elif backend is not None:
                connector = getattr(backend, "_connector", None)
            if connector and hasattr(connector, "close"):
                connector.close()
        except Exception:
            logger.error("Failed to close graph connector", exc_info=True)
        await super().stop()

    async def __aenter__(self) -> "MemoryService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
