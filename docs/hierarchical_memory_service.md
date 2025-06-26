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
