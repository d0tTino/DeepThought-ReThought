import json
import logging
from datetime import datetime, timezone
from typing import Any, List

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..graph import GraphDAL
from ..memory.hierarchical import HierarchicalMemory

logger = logging.getLogger(__name__)


class HierarchicalService:
    """Service combining vector search and graph lookups."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        vector_store: Any,
        graph_dal: GraphDAL,
        top_k: int = 3,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._memory = HierarchicalMemory(vector_store, graph_dal, top_k)

    def retrieve_context(self, prompt: str) -> List[str]:
        """Return context from vector store and graph."""
        return self._memory.retrieve_context(prompt)

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
            logger.info("HierarchicalService received input event ID %s", input_id)

            facts = self.retrieve_context(user_input)
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={
                    "retrieved_knowledge": {"facts": facts, "source": "hierarchical_service"}
                },
                input_id=input_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            await self._publisher.publish(
                EventSubjects.MEMORY_RETRIEVED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            logger.info("HierarchicalService published memory event ID %s", input_id)
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

        except Exception as e:  # pragma: no cover - defensive
            logger.error("Error in HierarchicalService handler: %s", e, exc_info=True)
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

    async def start(self, durable_name: str = "hierarchical_service_listener") -> bool:
        """Start listening for input events."""
        if self._subscriber is None:
            logger.error("Subscriber not initialized for HierarchicalService.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.INPUT_RECEIVED,
                handler=self._handle_input,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("HierarchicalService subscribed to %s", EventSubjects.INPUT_RECEIVED)
            return True
        except nats.errors.Error as e:
            logger.error("HierarchicalService failed to subscribe: %s", e, exc_info=True)
            return False
        except Exception as e:  # pragma: no cover - network failure
            logger.error("HierarchicalService failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop(self) -> None:
        """Stop listening for events."""
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("HierarchicalService stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
