# Hierarchical Memory Service

This document outlines the design of the experimental hierarchical memory system used in DeepThought-ReThought. The goal is to combine several storage backends so the agent can recall both recent interactions and long-term knowledge.

## Memory Layers

1. **BasicMemory** – thin wrapper over `TieredMemory` using an in-memory vector store.
2. **GraphMemory** – wraps `TieredMemory` with a file-backed graph database.
3. **KnowledgeGraphMemory** – persists structured facts in Memgraph using the GraphDAL layer.

The `CognitiveCoreService` coordinates these layers through a single `TieredMemory` instance.
When an `INPUT_RECEIVED` event arrives the service stores the text and publishes a
`MEMORY_RETRIEVED` event with the combined context.

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

Set the configuration keys used by GraphDAL when connecting to Memgraph:

```bash
export DT_MG_HOST=localhost
export DT_MG_PORT=7687
export DT_MG_USER=memgraph
export DT_MG_PASSWORD=memgraph
```

With these services running you can start your application and the memory service will connect automatically as long as it receives the proper NATS events.

## Exporting the Graph

After evaluating an interaction trace with `tools/replay.py` you may want to
inspect the knowledge graph. The `CognitiveCoreService` exposes a
`dump_graph(path)` method that writes the current graph in DOT format. Provide a
directory where the `graph.dot` file should be created:

```python
from deepthought.services import CognitiveCoreService

# use the configured vector backend ("chroma" or "faiss")
service = CognitiveCoreService.from_chroma(
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

The `CognitiveCoreService` now relies on the `TieredMemory` layer for context retrieval. Create a `TieredMemory` instance and pass it to the service or use `CognitiveCoreService.from_chroma()` which constructs one automatically using the configured vector backend.

## Offline Search

You can optionally attach a lightweight document index (e.g. a local Wikipedia dump).
Create a SQLite FTS index and point the service at the database file:

```python
from deepthought.search import OfflineSearch

search = OfflineSearch.create_index(
    "wiki.db",
    [("Title1", "Article text..."), ("Title2", "More text...")],
)
service = CognitiveCoreService(DummyNATS(), DummyJS(), memory, search=search)
```

Set ``DT_SEARCH_DB`` in your configuration file to load the index automatically.

Example ``config.yaml``:

```yaml
search_db: wiki.db
```

### Vector Backend

Set ``DT_VECTOR_BACKEND`` to ``faiss`` to use the in-memory FAISS store instead of Chroma.
When using FAISS, ``DT_VECTOR_USE_GPU`` enables GPU acceleration if available.

These environment variables correspond to the ``vector_backend``, ``vector_use_gpu``
and ``graph_backend`` fields of ``deepthought.config.Settings``. They may also be
set in a configuration file.

``DT_GRAPH_BACKEND`` selects the knowledge graph backend. Supported values include
``memgraph``, ``neo4j`` and ``noop``. These variables are read by
``CognitiveCoreService.from_chroma`` when initializing the backends.

### Graph Backend

``CognitiveCoreService.from_chroma`` takes a ``graph_backend_name`` string such as
``"memgraph"`` or ``"noop"``. The default connects to Memgraph using the
configuration keys ``DT_MG_HOST``, ``DT_MG_PORT``, ``DT_MG_USER`` and
``DT_MG_PASSWORD``. Pass ``"noop"`` to disable graph persistence during testing.

### Running with Neo4j

To try the service with Neo4j instead of Memgraph start a Neo4j container:

```bash
docker run --rm -p 7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:5
```

Set the corresponding variables so ``create_graph_backend("neo4j")`` can connect:

```bash
export DT_NEO4J_HOST=localhost
export DT_NEO4J_PORT=7687
export DT_NEO4J_USER=neo4j
export DT_NEO4J_PASSWORD=test
```

Then pass ``graph_backend_name="neo4j"`` to ``CognitiveCoreService.from_chroma``.

