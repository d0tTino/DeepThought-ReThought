import json
import logging
import os
from datetime import datetime, timezone
from typing import List

import networkx as nx
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class GraphMemory:
    """Graph-based memory using NetworkX persisted to a JSON file."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext, graph_file: str = "graph_memory.json"):
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._graph_file = graph_file

        dir_path = os.path.dirname(self._graph_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if os.path.exists(self._graph_file):
            self._graph = self._read_graph()
        else:
            self._graph = nx.DiGraph()
            self._write_graph()
        logger.info("GraphMemory initialized with file %s", self._graph_file)

    def _read_graph(self) -> nx.DiGraph:
        try:
            with open(self._graph_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return nx.readwrite.json_graph.node_link_graph(data)
        except Exception as e:
            logger.error(
                "Failed to read graph file %s: %s", self._graph_file, e, exc_info=True
            )
            empty_graph = nx.DiGraph()
            try:
                with open(self._graph_file, "w", encoding="utf-8") as f:
                    json.dump(
                        nx.readwrite.json_graph.node_link_data(empty_graph), f
                    )
            except Exception as write_err:
                logger.error(
                    "Failed to write empty graph file %s: %s",
                    self._graph_file,
                    write_err,
                    exc_info=True,
                )
            return empty_graph

    def _write_graph(self) -> None:
        data = nx.readwrite.json_graph.node_link_data(self._graph)
        with open(self._graph_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _add_interaction(self, user_input: str) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        node_id = timestamp
        self._graph.add_node(node_id, user_input=user_input, timestamp=timestamp)
        # Link to previous node if any
        nodes = list(self._graph.nodes(data=True))
        if len(nodes) > 1:
            prev_id = nodes[-2][0]
            self._graph.add_edge(prev_id, node_id, relation="next")
        self._write_graph()
        return node_id

    def _get_recent_facts(self, count: int = 3) -> List[str]:
        nodes = sorted(self._graph.nodes(data=True), key=lambda n: n[1].get("timestamp", ""))
        recent = nodes[-count:]
        return [n[1].get("user_input", "") for n in recent]

    async def _handle_input_event(self, msg: Msg) -> None:
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            input_id = data.get("input_id", "unknown")
            user_input = data.get("user_input", "")
            logger.info("GraphMemory received input event ID %s", input_id)

            self._add_interaction(user_input)
            facts = self._get_recent_facts()
            memory_data = {"facts": facts, "source": "graph_memory"}
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={"retrieved_knowledge": memory_data},
                input_id=input_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            await self._publisher.publish(EventSubjects.MEMORY_RETRIEVED, payload, use_jetstream=True, timeout=10.0)
            logger.info("GraphMemory published memory event ID %s", input_id)
            await msg.ack()
        except Exception as e:
            logger.error("Error in GraphMemory handler: %s", e, exc_info=True)
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

    async def start_listening(self, durable_name: str = "memory_graph_listener") -> bool:
        if not self._subscriber:
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
        except Exception as e:
            logger.error("GraphMemory failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop_listening(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("GraphMemory stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
