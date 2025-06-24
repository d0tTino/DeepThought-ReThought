# GraphDAL and Memory Service

GraphDAL is a lightweight data access layer powering `KnowledgeGraphMemory`. It connects to a running Memgraph instance through the `GraphConnector` class.

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

## Example Memory Service

The snippet below starts `KnowledgeGraphMemory` which listens for `INPUT_RECEIVED` events and stores them in Memgraph via GraphDAL:

```bash
python - <<'PY'
import asyncio, os
from nats.aio.client import Client as NATS
from deepthought.graph import GraphConnector
from deepthought.modules import KnowledgeGraphMemory

async def main():
    nc = NATS()
    await nc.connect(servers=[os.getenv("NATS_URL", "nats://localhost:4222")])
    js = nc.jetstream()
    connector = GraphConnector(
        host=os.getenv("MG_HOST", "localhost"),
        port=int(os.getenv("MG_PORT", "7687")),
        username=os.getenv("MG_USER", ""),
        password=os.getenv("MG_PASSWORD", ""),
    )
    memory = KnowledgeGraphMemory(nc, js, connector)
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
