import json
import logging
from datetime import datetime, timezone
from typing import Any, List, Sequence, Optional

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..graph import GraphDAL
from ..memory import create_vector_store

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
        self._vector_store = vector_store
        self._graph_dal = graph_dal
        self._top_k = top_k

    def _vector_matches(self, prompt: str) -> List[str]:
        if self._vector_store is None:
            return []
        try:
            result = self._vector_store.query(
                query_texts=[prompt], n_results=self._top_k
            )
            docs: Sequence | None = None
            if isinstance(result, dict):
                docs = result.get("documents")
            elif isinstance(result, Sequence):
                docs = result
            if not docs:
                return []
            matches: List[str] = []
            for doc in docs:
                if isinstance(doc, list):
                    for d in doc:
                        matches.append(str(getattr(d, "page_content", d)))
                else:
                    matches.append(str(getattr(doc, "page_content", doc)))
            return matches
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Vector store query failed: %s", exc, exc_info=True)
            return []

    def _graph_facts(self) -> List[str]:
        try:
            rows = self._graph_dal.query_subgraph(
                "MATCH (n:Entity) RETURN n.name AS fact LIMIT $limit",
                {"limit": self._top_k},
            )
            return [str(r.get("fact")) for r in rows if r.get("fact")]
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Graph query failed: %s", exc, exc_info=True)
            return []

    @classmethod
    def from_chroma(
        cls,
        nats_client: NATS,
        js_context: JetStreamContext,
        graph_dal: GraphDAL,
        collection_name: str = "deepthought",
        persist_directory: Optional[str] = None,
        top_k: int = 3,
    ) -> "HierarchicalService":
        """Instantiate with a new :class:`VectorStore` using Chroma."""
        store = create_vector_store(collection_name, persist_directory)
        return cls(nats_client, js_context, store, graph_dal, top_k)

    def retrieve_context(self, prompt: str) -> List[str]:
        """Return merged vector matches and graph facts."""
        vector = self._vector_matches(prompt)
        graph = self._graph_facts()
        seen = set()
        merged: List[str] = []
        for item in vector + graph:
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

            facts = self.retrieve_context(user_input)
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={
                    "retrieved_knowledge": {
                        "facts": facts,
                        "source": "hierarchical_service",
                    }
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
            logger.info(
                "HierarchicalService subscribed to %s", EventSubjects.INPUT_RECEIVED
            )
            return True
        except nats.errors.Error as e:
            logger.error(
                "HierarchicalService failed to subscribe: %s", e, exc_info=True
            )
            return False
        except Exception as e:  # pragma: no cover - network failure
            logger.error(
                "HierarchicalService failed to subscribe: %s", e, exc_info=True
            )
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
            "MATCH (a)-[r]->(b) RETURN id(a) AS src_id, coalesce(a.name, '') AS src, "
            "type(r) AS rel, id(b) AS dst_id, coalesce(b.name, '') AS dst",
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
