import json
import logging
import os
from datetime import datetime, timezone
from typing import List

import nats
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
        self.repaired = False

        dir_path = os.path.dirname(self._graph_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if os.path.exists(self._graph_file):
            self._graph, valid = self._read_graph()
            if not valid:
                self.repaired = True
                try:
                    self._write_graph()
                except Exception:
                    # _write_graph already logs the error
                    raise

        else:
            self._graph = nx.DiGraph()
            try:
                self._write_graph()
            except Exception:
                # _write_graph already logs the error
                raise
        logger.info("GraphMemory initialized with file %s", self._graph_file)

    def _read_graph(self) -> tuple[nx.DiGraph, bool]:

        try:
            with open(self._graph_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return nx.readwrite.json_graph.node_link_graph(data), True
        except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError) as e:
            self._last_read_error = e
            logger.error("Failed to read graph file %s: %s", self._graph_file, e, exc_info=True)
            return nx.DiGraph(), False

        except Exception as e:  # fallback
            self._last_read_error = e
            logger.error("Unexpected error reading graph file %s: %s", self._graph_file, e, exc_info=True)
            return nx.DiGraph(), False

    def _write_graph(self) -> None:
        data = nx.readwrite.json_graph.node_link_data(self._graph)
        try:
            with open(self._graph_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error("Failed to write graph file %s: %s", self._graph_file, e, exc_info=True)
            raise

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
            if not isinstance(data, dict):
                raise ValueError("InputReceived payload must be a dict")
            input_id = data.get("input_id")
            user_input = data.get("user_input")
            if not isinstance(input_id, str) or not isinstance(user_input, str):
                raise ValueError("Invalid input payload fields")
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
