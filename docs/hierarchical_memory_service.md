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
from deepthought.services import HierarchicalService

# use the configured vector backend ("chroma" or "faiss")
service = HierarchicalService.from_chroma(
    DummyNATS(),
    DummyJS(),
    graph_backend_name="memgraph",
    backend="faiss",
)
service.dump_graph("./graph_exports")
```

You can visualize the resulting DOT file with Graphviz:

```bash
dot -Tpng graph_exports/graph.dot -o graph.png
```

## Migration to TieredMemory

The `HierarchicalService` now relies on the `TieredMemory` layer for context retrieval. Create a `TieredMemory` instance and pass it to the service or use `HierarchicalService.from_chroma()` which constructs one automatically using the configured vector backend.

## Offline Search

You can optionally attach a lightweight document index (e.g. a local Wikipedia dump).
Create a SQLite FTS index and point the service at the database file:

```python
from deepthought.search import OfflineSearch

search = OfflineSearch.create_index(
    "wiki.db",
    [("Title1", "Article text..."), ("Title2", "More text...")],
)
service = HierarchicalService(DummyNATS(), DummyJS(), memory, search=search)
```

Set ``DT_SEARCH_DB`` in your configuration file to load the index automatically.

Example ``config.yaml``:

```yaml
search_db: wiki.db
```

### Vector Backend

Set ``DT_VECTOR_BACKEND`` to ``faiss`` to use the in-memory FAISS store instead of Chroma.
When using FAISS, ``DT_VECTOR_USE_GPU`` enables GPU acceleration if available.

### Graph Backend

``HierarchicalService.from_chroma`` takes a ``graph_backend_name`` string such as
``"memgraph"`` or ``"noop"``. The default connects to Memgraph using the
environment variables ``MG_HOST``, ``MG_PORT``, ``MG_USER`` and
``MG_PASSWORD``. Pass ``"noop"`` to disable graph persistence during testing.

### Running with Neo4j

To try the service with Neo4j instead of Memgraph start a Neo4j container:

```bash
docker run --rm -p 7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:5
```

Set the corresponding variables so ``create_graph_backend("neo4j")`` can connect:

```bash
export NEO4J_HOST=localhost
export NEO4J_PORT=7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=test
```

Then pass ``graph_backend_name="neo4j"`` to ``HierarchicalService.from_chroma``.

