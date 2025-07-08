# GraphDAL and Memory Service

GraphDAL is a lightweight data access layer powering `KnowledgeGraphMemory`. It connects to a running graph database through the `GraphConnector` class. Both Neo4j and Memgraph are supported backends.

## Starting Memgraph

Launch Memgraph using Docker:

```bash
docker run --rm -p 7687:7687 memgraph/memgraph
```

## Environment Variables

Set the following variables so the memory service can reach Memgraph:

| Variable       | Description           | Default     |
| -------------- | -------------------- | ----------- |
| `MG_HOST`      | Memgraph host name   | `localhost` |
| `MG_PORT`      | Memgraph port number | `7687`      |
| `MG_USER`      | Username (optional)  | *(empty)*   |
| `MG_PASSWORD`  | Password (optional)  | *(empty)*   |

## Starting Neo4j

Run a local Neo4j instance with Docker:

```bash
docker run --rm -p 7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:5
```

Set these variables so ``create_graph_backend("neo4j")`` can connect:

| Variable         | Description           | Default     |
| ---------------- | -------------------- | ----------- |
| `NEO4J_HOST`     | Neo4j host name      | `localhost` |
| `NEO4J_PORT`     | Neo4j port number    | `7687`      |
| `NEO4J_USER`     | Username             | `neo4j`     |
| `NEO4J_PASSWORD` | Password             | `neo4j`     |

Both connectors read these environment variables automatically when no
explicit parameters are provided and will retry connecting a few times
before failing.

## Example Memory Service

The snippet below starts `KnowledgeGraphMemory` which listens for `INPUT_RECEIVED` events and stores them in Memgraph via GraphDAL:

```bash
python - <<'PY'
import asyncio, os
from nats.aio.client import Client as NATS
from deepthought.graph import GraphConnector, GraphDAL
from deepthought.modules import KnowledgeGraphMemory

async def main():
    nc = NATS()
    await nc.connect(servers=[os.getenv("NATS_URL", "nats://localhost:4222")])
    js = nc.jetstream()
    connector = GraphConnector()  # reads MG_* variables by default
    dal = GraphDAL(connector)
    memory = KnowledgeGraphMemory(nc, js, dal)
    await memory.start_listening()
    await asyncio.Event().wait()

asyncio.run(main())
PY
```

This service can be used alongside other modules to persist conversations in a graph database.

Alternatively, run the example script directly:

```bash
python examples/memgraph_memory_service.py
```

Another demo using a Neo4j backend is available:

```bash
python examples/graph_memory_demo.py
```


## Hierarchical Memory Wrapper

`HierarchicalMemory` combines vector search (using a store like Chroma) with
graph lookups through `GraphDAL`. Instantiate it with a vector store and an
existing `GraphDAL` instance:

```python
from deepthought.memory.hierarchical import HierarchicalMemory
from deepthought.graph import GraphConnector, GraphDAL
import chromadb

# Initialize vector store and graph connection
client = chromadb.Client()
collection = client.create_collection("my_vectors")
connector = GraphConnector()
dal = GraphDAL(connector)

memory = HierarchicalMemory(collection, dal)
context = memory.retrieve_context("Where was I yesterday?")
print(context)
```

`retrieve_context()` returns a list of strings merging the top matches from the
vector store with recent facts from the graph.
