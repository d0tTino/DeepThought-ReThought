import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Sequence

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.errors import Error as NatsError
from nats.js.client import JetStreamContext

from ..config import get_settings
from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..graph import GraphDAL
from ..memory.tiered import TieredMemory
from ..memory.vector_store import create_vector_store
from ..search import OfflineSearch

logger = logging.getLogger(__name__)


class HierarchicalService:
    """Service combining vector search and graph lookups."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        memory: TieredMemory | None,
        search: OfflineSearch | None = None,
        graph_dal: GraphDAL | None = None,
        top_k: int = 3,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._memory = memory
        if search is None:
            settings = get_settings()
            db_path = settings.search_db
            if db_path:
                if not os.path.exists(db_path):
                    search = OfflineSearch.create_index(db_path, [])
                else:
                    search = OfflineSearch(db_path)
        self._search = search
        self._top_k = top_k

    def _vector_matches(self, prompt: str) -> List[str]:
        """Return vector matches using the underlying memory store."""
        return self._memory._vector_matches(prompt)

    def _graph_facts(self) -> List[str]:
        """Return graph facts using the underlying memory store."""
        return self._memory._graph_facts(self._memory._top_k)

    @classmethod
    def from_chroma(
        cls,
        nats_client: NATS,
        js_context: JetStreamContext,
        graph_dal: GraphDAL,
        collection_name: str = "deepthought",
        persist_directory: Optional[str] = None,
        capacity: int = 100,
        top_k: int = 3,
        search_db: Optional[str] = None,
    ) -> "HierarchicalService":
        """Instantiate with a new :class:`TieredMemory` using Chroma."""
        store = create_vector_store(collection_name, persist_directory)
        memory = TieredMemory(store, graph_dal, capacity=capacity, top_k=top_k)
        db_path = search_db or get_settings().search_db
        if db_path:
            if not os.path.exists(db_path):
                search = OfflineSearch.create_index(db_path, [])
            else:
                search = OfflineSearch(db_path)
        else:
            search = None
        return cls(nats_client, js_context, memory, search=search, top_k=top_k)

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
                except NatsError:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except NatsError:
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

    def dump_graph(self, path: str) -> str:
        """Write the underlying graph to ``path`` in DOT format."""
        import os

        os.makedirs(path, exist_ok=True)
        dot_path = os.path.join(path, "graph.dot")

        rows = self._memory._dal.query_subgraph(
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
