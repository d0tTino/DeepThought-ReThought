import json
import logging
import os
from datetime import datetime, timezone
from typing import List

import networkx as nx

logger = logging.getLogger(__name__)


class GraphDAL:
    """Simple file-backed graph storage using NetworkX."""

    def __init__(self, graph_file: str = "graph_memory.json") -> None:
        self._graph_file = graph_file
        dir_path = os.path.dirname(graph_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if os.path.exists(graph_file):
            self._graph = self._read_graph()
        else:
            self._graph = nx.DiGraph()
            self._write_graph()
        logger.info("GraphDAL initialized with file %s", graph_file)

    def _read_graph(self) -> nx.DiGraph:
        try:
            with open(self._graph_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return nx.readwrite.json_graph.node_link_graph(data)
        except Exception as e:
            logger.error("Failed to read graph file %s: %s", self._graph_file, e, exc_info=True)
            return nx.DiGraph()

    def _write_graph(self) -> None:
        data = nx.readwrite.json_graph.node_link_data(self._graph)
        with open(self._graph_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def add_interaction(self, user_input: str) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        node_id = timestamp
        self._graph.add_node(node_id, user_input=user_input, timestamp=timestamp)
        nodes = list(self._graph.nodes())
        if len(nodes) > 1:
            self._graph.add_edge(nodes[-2], node_id, relation="next")
        self._write_graph()
        return node_id

    def get_recent_facts(self, count: int = 3) -> List[str]:
        nodes = sorted(self._graph.nodes(data=True), key=lambda n: n[1].get("timestamp", ""))
        recent = nodes[-count:]
        return [n[1].get("user_input", "") for n in recent]
