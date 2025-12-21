"""Knowledge graph memory module using a Memgraph backend."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext
from rdflib.namespace import RDF

from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..graph import GraphDAL
from ..ontology import OntologyManager

logger = logging.getLogger(__name__)


class KnowledgeGraphMemory:
    """Parse user input into a graph stored in Memgraph."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        dal: GraphDAL,
        ontology: Optional[OntologyManager] = None,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._dal = dal
        self._ontology = ontology

    def _parse_input(self, text: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        words = [w for w in text.strip().split() if w]
        nodes = list(dict.fromkeys(words))
        edges = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        return nodes, edges

    def _store(self, nodes: List[str], edges: List[Tuple[str, str]]) -> None:
        for name in nodes:
            self._dal.merge_entity(name)
        for src, dst in edges:
            self._dal.merge_next_edge(src, dst)
        if self._ontology is not None:
            from rdflib import Namespace
            from rdflib.namespace import OWL

            ex = Namespace("http://deepthought.local/graph#")
            for name in nodes:
                self._ontology.add_triples(
                    [
                        (ex[name], RDF.type, OWL.NamedIndividual),
                        (ex[name], RDF.type, ex.Entity),
                    ]
                )
            for src, dst in edges:
                self._ontology.add_triple(ex[src], ex.next, ex[dst])

    def infer_facts(self) -> List[Tuple[str, str, str]]:
        """Return inferred triples from the ontology."""
        if self._ontology is None:
            return []
        return self._ontology.infer_facts()

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
            logger.info("KnowledgeGraphMemory received input %s", input_id)

            nodes, edges = self._parse_input(user_input)
            self._store(nodes, edges)

            facts = self.infer_facts()
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={"facts": facts, "source": "knowledge_graph"},
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
            await msg.ack()
        except (
            json.JSONDecodeError,
            ValueError,
        ) as e:  # pragma: no cover - validation errors
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

        except Exception as e:  # pragma: no cover - error path
            logger.error("Error in KnowledgeGraphMemory: %s", e, exc_info=True)
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

    async def start_listening(
        self, durable_name: str = "knowledge_graph_listener"
    ) -> bool:
        if not self._subscriber:
            logger.error("Subscriber not initialized for KnowledgeGraphMemory.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.INPUT_RECEIVED,
                handler=self._handle_input_event,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info(
                "KnowledgeGraphMemory subscribed to %s", EventSubjects.INPUT_RECEIVED
            )
            return True
        except nats.errors.Error as e:  # pragma: no cover - network failure
            logger.error(
                "KnowledgeGraphMemory failed to subscribe: %s", e, exc_info=True
            )
            return False
        except Exception as e:  # pragma: no cover - network failure
            logger.error(
                "KnowledgeGraphMemory failed to subscribe: %s", e, exc_info=True
            )
            return False

    async def stop_listening(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("KnowledgeGraphMemory stopped listening.")
        else:  # pragma: no cover - defensive
            logger.warning("Cannot stop listening - no subscriber available.")
