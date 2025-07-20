import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Sequence

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.errors import Error as NatsError
from nats.js.client import JetStreamContext

from ..config import Settings
from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..memory import create_memory_backend
from ..memory.tiered import TieredMemory
from ..metrics.prometheus import INPUT_LATENCY_SECONDS, INPUTS_TOTAL
from ..search import OfflineSearch

logger = logging.getLogger(__name__)


class HierarchicalService:
    """Service combining vector search and graph lookups."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        settings: Settings,
        memory: TieredMemory | None = None,
        search: OfflineSearch | None = None,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._nc = nats_client
        self._memory = memory
        if self._memory is None:
            self._memory = create_memory_backend(settings=settings)
        if search is None:
            db_path = settings.search_db
            if db_path:
                if not os.path.exists(db_path):
                    search = OfflineSearch.create_index(db_path, [])
                else:
                    search = OfflineSearch(db_path)
        self._search = search
        self._top_k = settings.memory_top_k

    def _vector_matches(self, prompt: str) -> List[str]:
        """Return vector matches using the underlying memory store."""
        return self._memory.vector_matches(prompt)

    def _graph_facts(self) -> List[str]:
        """Return graph facts using the underlying memory store."""
        return self._memory.graph_facts()

    @classmethod
    def from_chroma(
        cls,
        nats_client: NATS,
        js_context: JetStreamContext,
        settings: Settings,
    ) -> "HierarchicalService":
        """Instantiate with a new :class:`TieredMemory` using Chroma."""
        memory = create_memory_backend(settings=settings)
        db_path = settings.search_db
        if db_path:
            if not os.path.exists(db_path):
                try:
                    search = OfflineSearch.create_index(db_path, [])
                except ValueError:
                    logger.warning("No documents available for search index; disabling offline search")
                    search = None
            else:
                search = OfflineSearch(db_path)
        else:
            search = None
        return cls(nats_client, js_context, settings, memory, search=search)

    def retrieve_context(self, prompt: str) -> List[str]:
        """Return retrieved facts using :class:`TieredMemory` and optional search."""
        memory_facts = self._memory.retrieve_context(prompt) if self._memory else []
        search_facts: List[str] = []
        if self._search:
            try:
                search_facts = self._search.search(prompt, limit=self._top_k)
            except Exception:  # pragma: no cover - defensive
                logger.error("Offline search failed", exc_info=True)
        seen = set()
        merged: List[str] = []
        for item in memory_facts + search_facts:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged

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
            logger.info("HierarchicalService received input event ID %s", input_id)

            facts: Sequence[str] = self.retrieve_context(user_input)
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={
                    "facts": facts,
                    "source": "hierarchical_service",
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

        except Exception as e:  # pragma: no cover - defensive
            logger.error("Error in HierarchicalService handler: %s", e, exc_info=True)
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
            INPUTS_TOTAL.labels(service="hierarchical_service").inc()
            INPUT_LATENCY_SECONDS.labels(service="hierarchical_service").observe(duration)

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
        except NatsError as e:

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
        if getattr(self, "_nc", None) and getattr(self._nc, "is_connected", False):
            await self._nc.drain()

    async def __aenter__(self) -> "HierarchicalService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    def dump_graph(self, path: str) -> str:
        """Write the underlying graph to ``path`` in DOT format."""
        import os

        if self._memory is None:
            raise ValueError("Cannot dump graph without memory")

        os.makedirs(path, exist_ok=True)
        dot_path = os.path.join(path, "graph.dot")

        rows = self._memory.graph_backend.query_subgraph(
            (
                "MATCH (a)-[r]->(b) RETURN id(a) AS src_id, "
                "coalesce(a.name, '') AS src, type(r) AS rel, "
                "id(b) AS dst_id, coalesce(b.name, '') AS dst"
            ),
            {},
        )

        seen = set()
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write("digraph {\n")
            for row in rows:
                src = row.get("src") or f"node{row.get('src_id')}"
                dst = row.get("dst") or f"node{row.get('dst_id')}"
                if src not in seen:
                    f.write(f'    "{src}";\n')
                    seen.add(src)
                if dst not in seen:
                    f.write(f'    "{dst}";\n')
                    seen.add(dst)
                rel = row.get("rel", "")
                f.write(f'    "{src}" -> "{dst}" [label="{rel}"];\n')
            f.write("}\n")
        logger.info("Graph dumped to %s", dot_path)
        return dot_path
