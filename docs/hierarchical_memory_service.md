# Hierarchical Memory Service

This document outlines the design of the experimental hierarchical memory system used in DeepThought-ReThought. The goal is to combine several storage backends so the agent can recall both recent interactions and long-term knowledge.

## Memory Layers

1. **BasicMemory** – stores recent messages in a JSON file on disk.
2. **VectorMemory** – persists embeddings in a Chroma database for semantic search.
3. **KnowledgeGraphMemory** – persists structured facts in Memgraph using the GraphDAL layer.

The `MemoryService` coordinates these layers. When an `INPUT_RECEIVED` event arrives the service updates each layer and aggregates their retrieved facts into a single `MEMORY_RETRIEVED` event.

## Running the Service

Make sure the following services are available before starting the memory service:

### Chroma

```bash
docker run --rm -p 8000:8000 chromadb/chroma
```

### Memgraph

```bash
docker run --rm -p 7687:7687 memgraph/memgraph
```

Set the environment variables used by GraphDAL when connecting to Memgraph:

```bash
export MG_HOST=localhost
export MG_PORT=7687
export MG_USER=memgraph
export MG_PASSWORD=memgraph
```

With these services running you can start your application and the memory service will connect automatically as long as it receives the proper NATS events.

## Exporting the Graph

After evaluating an interaction trace with `tools/replay.py` you may want to
inspect the knowledge graph. The `HierarchicalService` exposes a
`dump_graph(path)` method that writes the current graph in DOT format. Provide a
directory where the `graph.dot` file should be created:

```python
from deepthought.graph import GraphConnector, GraphDAL
from deepthought.memory import TieredMemory
from deepthought.services import HierarchicalService

connector = GraphConnector(host="localhost", port=7687)
dal = GraphDAL(connector)
memory = TieredMemory.from_chroma(dal)
service = HierarchicalService(DummyNATS(), DummyJS(), memory)
service.dump_graph("./graph_exports")
```

You can visualize the resulting DOT file with Graphviz:

```bash
dot -Tpng graph_exports/graph.dot -o graph.png
```

## Migration to TieredMemory

The `HierarchicalService` now relies on the `TieredMemory` layer for context retrieval. Create a `TieredMemory` instance and pass it to the service or use `HierarchicalService.from_chroma()` which constructs one automatically.

