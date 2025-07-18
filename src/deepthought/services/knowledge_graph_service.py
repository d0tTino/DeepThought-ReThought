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
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..graph import GraphConnector, GraphDAL, GraphDALBackend, Neo4jConnector
from ..memory.tiered import TieredMemory
from ..memory.vector_store import create_vector_store

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """Service persisting interactions in a graph database via GraphDAL."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        settings: Settings | None = None,
        memory: Optional[TieredMemory] = None,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._nc = nats_client
        self._settings = settings or get_settings()
        if memory is None:
            store = create_vector_store(
                backend=self._settings.vector_backend,
                use_gpu=self._settings.vector_use_gpu,
            )
            if self._settings.graph_backend.lower() == "neo4j":
                connector = Neo4jConnector(
                    host=self._settings.neo4j_host,
                    port=self._settings.neo4j_port,
                    username=self._settings.neo4j_user,
                    password=self._settings.neo4j_password,
                )
            else:
                connector = GraphConnector(
                    host=self._settings.mg_host,
                    port=self._settings.mg_port,
                    username=self._settings.mg_user,
                    password=self._settings.mg_password,
                )
            backend = GraphDALBackend(GraphDAL(connector))
            memory = TieredMemory(
                store,
                backend,
                capacity=self._settings.memory_capacity,
                top_k=self._settings.memory_top_k,
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
            logger.info("KnowledgeGraphService received input %s", input_id)

            self._memory.store_interaction(user_input)
            facts = self._memory.retrieve_context(user_input)
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={"facts": facts, "source": "knowledge_graph"},
                input_id=input_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            await self._publisher.publish(
                EventSubjects.MEMORY_RETRIEVED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            logger.info("KnowledgeGraphService published memory event %s", input_id)
            if hasattr(msg, "ack") and callable(msg.ack):
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
            logger.error("Error in KnowledgeGraphService: %s", e, exc_info=True)
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
            logger.info("KnowledgeGraphService processed input in %.3fs", duration)

    async def start(self, durable_name: str = "knowledge_graph_listener") -> bool:
        if self._subscriber is None:
            logger.error("Subscriber not initialized for KnowledgeGraphService.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.INPUT_RECEIVED,
                handler=self._handle_input,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("KnowledgeGraphService subscribed to %s", EventSubjects.INPUT_RECEIVED)
            return True
        except Exception as e:  # pragma: no cover - network failure
            logger.error("KnowledgeGraphService failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("KnowledgeGraphService stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
        if getattr(self, "_nc", None) and getattr(self._nc, "is_connected", False):
            await self._nc.drain()
