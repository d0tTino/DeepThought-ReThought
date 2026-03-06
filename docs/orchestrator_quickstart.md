# Orchestrator Quick Start

The orchestrator starts multiple services with a single NATS connection. Each service is registered via an entry point in the `deepthought.services` group.

## Step-by-Step Setup

1. **Start NATS**

   Launch a local NATS server with JetStream enabled:

   ```bash
   ./scripts/start_nats.sh
   ```

   Then create the required stream:

   ```bash
   python -m setup_jetstream
   ```

2. **Choose a Graph Backend**

   Either Memgraph or Neo4j can be used. Export the appropriate environment
   variables so the `knowledge_graph` service can connect.

   *Memgraph*

   ```bash
   docker compose up -d memgraph
   export DT_GRAPH_BACKEND=memgraph
   export DT_MG_HOST=localhost
   export DT_MG_PORT=7687
   export DT_MG_USER=memgraph
   export DT_MG_PASSWORD=memgraph
   ```

   *Neo4j*

   ```bash
   docker compose up -d neo4j
   export DT_GRAPH_BACKEND=neo4j
   export DT_NEO4J_HOST=localhost
   export DT_NEO4J_PORT=7687
   export DT_NEO4J_USER=neo4j
   export DT_NEO4J_PASSWORD=test
   ```

3. **Write a configuration file**

   Create `orchestrator.yaml` listing the services to launch.
   Temporary crews or LangGraph graphs can be added using callables
   referenced as `module:function` strings:

   ```yaml
   services:
     - memory
     - knowledge_graph
     - discord_gateway
   crews:
     - examples.crew_demo:create_demo_crew
   graphs:
     - examples.multi_agent_demo:run_graph
   ```

   For a complete, canonical wiring map (subjects, durable consumer names,
   and environment variable references), copy from
   [`examples/orchestrator.yml`](../examples/orchestrator.yml).

   The default production DAG includes feedback adaptation: keep the `feedback` service enabled with durable subscriptions on `dtr.response.ranked.v1`, `dtr.feedback.outcome_signal.v1`, and `dtr.feedback.correction_signal.v1`.

   Set `DT_MEMORY_DB` and `DT_SOCIAL_GRAPH_DB` so adaptation writes are persisted, and optionally set `DT_FEEDBACK_DURABLE_PREFIX` to control durable naming (defaults to `feedback_service`).

   `llm_remote` requires the `LLM_ENDPOINT` variable pointing to a running
   `/generate` endpoint.

   When `discord_gateway` is enabled, it subscribes to `dtr.response.ranked` and forwards
   final responses back to the originating Discord channel (using a durable consumer).

4. **Run the orchestrator**

   ```bash
   dtrt orchestrate orchestrator.yaml
   ```

Publishing an `INPUT_RECEIVED` event will now flow through the configured
services, demonstrating end-to-end event handling.

## Monitoring Services

Set `METRICS_PORT=8000` before starting the orchestrator to expose Prometheus
metrics. For a full dashboard, launch Prometheus and Grafana using the provided
Compose file:

```bash
docker compose -f docker-compose.metrics.yml up
```

Grafana will be available on <http://localhost:3000> with the default password
`admin`.

## End-to-End Demo

The `examples/end_to_end_demo.py` script shows how to run multiple services via
the orchestrator. It launches a local NATS server and starts the
`CognitiveCoreService`, `RemoteLLM`, `OutputHandler` and `RewardManager` services. A
sample message is then processed end-to-end.

Run the demo with:

```bash
python examples/end_to_end_demo.py
```

RemoteLLM requires the `LLM_ENDPOINT` environment variable to point to a
running `/generate` endpoint when not using the built-in edge server.
