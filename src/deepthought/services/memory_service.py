import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..memory import create_memory_backend
from ..memory.tiered import TieredMemory
from ..metrics.prometheus import INPUT_LATENCY_SECONDS, INPUTS_TOTAL

logger = logging.getLogger(__name__)


class MemoryService:
    """Service that stores and retrieves interactions using :class:`TieredMemory`."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        memory: Optional[TieredMemory] = None,
        *,
        graph_backend_name: str | None = None,
        collection_name: str = "deepthought",
        persist_directory: str | None = None,
        vector_backend: str | None = None,
        use_gpu: bool | None = None,
        capacity: int = 100,
        top_k: int = 3,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._nc = nats_client
        if memory is None:
            memory = create_memory_backend(
                graph_backend_name=graph_backend_name,
                collection_name=collection_name,
                persist_directory=persist_directory,
                vector_backend=vector_backend,
                use_gpu=use_gpu,
                capacity=capacity,
                top_k=top_k,
            )
        self._memory = memory

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
            logger.error("Error in MemoryService handler: %s", e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except nats.errors.Error:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except nats.errors.Error:
                    logger.error("Failed to ack message after error", exc_info=True)
        finally:
            duration = time.perf_counter() - start
            INPUTS_TOTAL.labels(service="memory_service").inc()
            INPUT_LATENCY_SECONDS.labels(service="memory_service").observe(duration)

    async def start(self, durable_name: str = "memory_service_listener") -> bool:
        if self._subscriber is None:
            logger.error("Subscriber not initialized for MemoryService.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.INPUT_RECEIVED,
                handler=self._handle_input,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("MemoryService subscribed to %s", EventSubjects.INPUT_RECEIVED)
            return True
        except nats.errors.Error as e:
            logger.error("MemoryService failed to subscribe: %s", e, exc_info=True)
            return False
        except Exception as e:
            logger.error("MemoryService failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("MemoryService stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
        if getattr(self, "_nc", None) and self._nc.is_connected:
            await self._nc.drain()
