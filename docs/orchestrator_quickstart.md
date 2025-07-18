# Orchestrator Quick Start

The orchestrator starts multiple services with a single NATS connection. Each service is registered via an entry point in the `deepthought.services` group.

Create a configuration file listing the services to launch:

```yaml
services:
  - demo
  - codegen
  - knowledge_graph
```

Set `DT_GRAPH_BACKEND` to either `memgraph` or `neo4j` and export the matching
connection variables (see `docs/graphdal.md`) so the service can reach your
database.

Start a local NATS server with JetStream enabled then run:

```bash
dtrt orchestrate orchestrator.yaml
```

Publishing an `INPUT_RECEIVED` event will now flow through the `DemoService` and the `CodeGenerationService`, demonstrating end-to-end event handling.

## End-to-End Demo

The `examples/end_to_end_demo.py` script shows how to run multiple services via
the orchestrator. It launches a local NATS server and starts the
`MemoryService`, `RemoteLLM`, `OutputHandler` and `RewardManager` services. A
sample message is then processed end-to-end.

Run the demo with:

```bash
python examples/end_to_end_demo.py
```
