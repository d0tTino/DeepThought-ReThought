# GraphDAL and Memory Service

GraphDAL is a lightweight data access layer powering `KnowledgeGraphMemory`. It connects to a running graph database through the `GraphConnector` class. Both Neo4j and Memgraph are supported backends.

## Starting Memgraph

Launch Memgraph using Docker:

```bash
docker run --rm -p 7687:7687 memgraph/memgraph
```

Alternatively start the container defined in `docker-compose.yml`:

```bash
docker compose up -d memgraph
```

## Environment Variables

Set the following variables so the memory service can reach Memgraph:

| Variable       | Description           | Default     |
| -------------- | -------------------- | ----------- |
| `DT_MG_HOST` / `MG_HOST` | Memgraph host name | `localhost` |
| `DT_MG_PORT`      | Memgraph port number | `7687`      |
| `DT_MG_USER`      | Username (optional)  | `memgraph`  |
| `DT_MG_PASSWORD`  | Password (optional)  | `memgraph`  |

## Starting Neo4j

Run a local Neo4j instance with Docker:

```bash
docker run --rm -p 7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:5
```

You can also start the Neo4j service via Compose:

```bash
docker compose up -d neo4j
```

Set these variables so ``create_graph_backend("neo4j")`` can connect:

| Variable         | Description           | Default     |
| ---------------- | -------------------- | ----------- |
| `DT_NEO4J_HOST`     | Neo4j host name      | `localhost` |
| `DT_NEO4J_PORT`     | Neo4j port number    | `7687`      |
| `DT_NEO4J_USER`     | Username             | `neo4j`     |
| `DT_NEO4J_PASSWORD` | Password             | `neo4j`     |

Both connectors read these environment variables automatically when no
explicit parameters are provided and will retry connecting a few times
before failing.

## Example Memory Service

Start the unified `MemoryService` which configures its backends from environment variables:

```bash
python - <<'PY'
import asyncio

from nats.aio.client import Client as NATS

from deepthought.config import get_settings
from deepthought.services import MemoryService


async def main():
    settings = get_settings()
    nc = NATS()
    await nc.connect(servers=[settings.nats_url])
    js = nc.jetstream()

    service = MemoryService.from_config(nc, js)
    await service.start()
    await asyncio.Event().wait()

asyncio.run(main())
PY
```

This service can be used alongside other modules to persist conversations in a graph database.

Alternatively, run the example script directly:

```bash
python examples/memgraph_memory_service.py
```

You can try a standalone graph memory demo once either Memgraph or Neo4j is
running. Choose the backend via ``DT_GRAPH_BACKEND`` before executing the
script. A ``noop`` backend is also available for quick testing without any
database:

```bash
# For Memgraph
export DT_GRAPH_BACKEND=memgraph
python examples/graph_memory_demo.py
# For Neo4j
export DT_GRAPH_BACKEND=neo4j
python examples/graph_memory_demo.py
# Or to quickly run it with no database
export DT_GRAPH_BACKEND=noop
python examples/graph_memory_demo.py
```

### Orchestrator Configuration

Start the `knowledge_graph` service via the orchestrator by writing an
`orchestrator.yml` file:

```yaml
services:
  - knowledge_graph
```

Choose the backend with `DT_GRAPH_BACKEND` and set the connection variables.
For Neo4j:

```bash
export DT_GRAPH_BACKEND=neo4j
export DT_NEO4J_HOST=localhost
export DT_NEO4J_PORT=7687
export DT_NEO4J_USER=neo4j
export DT_NEO4J_PASSWORD=test
```

For Memgraph simply set `DT_GRAPH_BACKEND=memgraph` and optionally the
`DT_MG_*` variables. Then run:

```bash
dtrt orchestrate orchestrator.yml
```


## Hierarchical Memory Wrapper

`HierarchicalMemory` combines vector search (using a store like Chroma) with
graph lookups through `GraphDAL`. Instantiate it with a vector store and an
existing `GraphDAL` instance:

```python
import chromadb

from deepthought.graph import GraphConnector, GraphDAL
from deepthought.memory.hierarchical import HierarchicalMemory

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
