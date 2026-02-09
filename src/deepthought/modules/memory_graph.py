import json
import logging
from datetime import datetime, timezone
from typing import Optional

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..memory.tiered import TieredMemory
from ..memory.vector_store import create_vector_store
from ..services.file_graph_dal import FileGraphBackend

logger = logging.getLogger(__name__)


class GraphMemory:
    """File-based graph memory backed by :class:`TieredMemory`."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        *,
        memory: Optional[TieredMemory] = None,
        graph_file: str = "graph_memory.json",
        capacity: int = 100,
        top_k: int = 3,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._nc = nats_client
        if memory is None:
            store = create_vector_store()
            backend = FileGraphBackend(graph_file)
            memory = TieredMemory(store, backend, capacity=capacity, top_k=top_k)
        self._memory = memory
        logger.info("GraphMemory initialized using TieredMemory with %s", graph_file)

    async def _handle_input_event(self, msg: Msg) -> None:
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("InputReceived payload must be a dict")
            input_id = data.get("input_id")
            user_input = data.get("user_input")
            user_id = data.get("user_id") or (msg.headers.get("user_id") if msg.headers else None)
            if not isinstance(user_id, str):
                user_id = None
            if not isinstance(input_id, str) or not isinstance(user_input, str):
                raise ValueError("Invalid input payload fields")
            logger.info("GraphMemory received input event ID %s", input_id)

            self._memory.store_interaction(user_input)
            facts = self._memory.retrieve_context(user_input)
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={"facts": facts, "source": "graph_memory"},
                user_input=user_input,
                input_id=input_id,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            await self._publisher.publish(
                EventSubjects.MEMORY_RETRIEVED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            logger.info("GraphMemory published memory event ID %s", input_id)
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
            logger.error("Error in GraphMemory handler: %s", e, exc_info=True)
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

    async def start_listening(self, durable_name: str = "memory_graph_listener") -> bool:
        if self._subscriber is None:
            logger.error("Subscriber not initialized for GraphMemory.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.INPUT_RECEIVED,
                handler=self._handle_input_event,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("GraphMemory subscribed to %s", EventSubjects.INPUT_RECEIVED)
            return True
        except nats.errors.Error as e:
            logger.error("GraphMemory failed to subscribe: %s", e, exc_info=True)
            return False
        except Exception as e:
            logger.error("GraphMemory failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop_listening(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("GraphMemory stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
        if getattr(self, "_nc", None) and self._nc.is_connected:
            await self._nc.drain()
