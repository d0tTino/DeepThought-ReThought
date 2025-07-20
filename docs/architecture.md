# System Architecture

This document provides a high level overview of how the main services in **DeepThought‑ReThought** interact and how to set up optional components like the FAISS vector store, FastAPI endpoints and the metrics system.

## Service Interactions

The project follows an event driven architecture built on NATS/JetStream. Components publish and subscribe to event subjects defined in `src/deepthought/eda/events.py`.

```mermaid
sequenceDiagram
    participant User
    participant Bot
    participant MemoryService
    participant HierarchicalService
    participant LLM

    User->>Bot: Message
    Bot->>NATS: INPUT_RECEIVED
    NATS->>MemoryService: INPUT_RECEIVED
    MemoryService->>NATS: MEMORY_RETRIEVED
    NATS->>HierarchicalService: MEMORY_RETRIEVED
    HierarchicalService->>LLM: Query
    LLM-->>HierarchicalService: Response
    HierarchicalService->>NATS: RESPONSE_GENERATED
    NATS->>Bot: RESPONSE_GENERATED
    Bot-->>User: Reply
```

The example Discord bot in `bot.py` sends `INPUT_RECEIVED` events, retrieves knowledge from memory services and ultimately receives a `RESPONSE_GENERATED` message containing the model output.

## FAISS Setup

`FaissVectorStore` in `src/deepthought/memory/vector_store.py` offers a lightweight vector database for similarity search. To enable it:

- Install FAISS and run the unit test:

  ```bash
  pip install faiss-cpu  # or faiss-gpu if available
  pytest tests/unit/test_faiss_vector_store.py
  ```
  The test output should report `1 passed` indicating FAISS is working.

- Create a store and query vectors:

  ```python
  from deepthought.memory.vector_store import FaissVectorStore

  store = FaissVectorStore()
  store.add_texts(["hello world", "goodbye"], ids=["1", "2"])
  results = store.query(["hello"], n_results=1)
  ```

## API Endpoints

`examples/prism_server.py` exposes a minimal FastAPI service used by the social graph bot.

- Launch the server with an authorization token:

  ```bash
  export PRISM_TOKENS=my-secret-token
  python examples/prism_server.py
  ```

- Send a request to the `/receive_data` endpoint:

  ```bash
  curl -X POST \
       -H "Authorization: Bearer my-secret-token" \
       -H "Content-Type: application/json" \
       -d '{"message": "hello"}' \
       http://localhost:5000/receive_data
  ```

Configure the bot with the `PRISM_ENDPOINT` environment variable to forward JSON payloads to this endpoint.

## Metrics System

Execution traces from bots or training runs can be replayed and analyzed.

- Send recorded events through NATS and store metrics:

  ```bash
  python tools/discord_replay.py traces.jsonl --metrics metrics.json
  ```

- Visualize the results with the dashboard utility:

  ```bash
  pip install matplotlib
  python tools/dashboard.py metrics.json --show
  ```

The generated `dashboard.png` illustrates BLEU, ROUGE‑L and average latency over time.
For a Prometheus and Grafana setup see [prometheus_grafana.md](prometheus_grafana.md).

## Service Template

A minimal skeleton is available under `deepthought/templates/service/`. It demonstrates
how to use the shared `Publisher` and `Subscriber` helpers.
The bus CLI exposes an initializer for creating new services.
To create a new service based on this template run:

```bash
dtrt init service <name>
```

The same initializer is available under the `bus` command with JetStream
persistence enabled by default:

```bash
dtrt bus init service <name>
```

This variant also creates an `nats.env.example` file alongside the
`Dockerfile` showing how to configure credentials and mTLS settings. The
`Publisher` and `Subscriber` helpers read the following variables:

```
NATS_URL=nats://localhost:4222
NATS_TLS_CERT=/path/to/client-cert.pem
NATS_TLS_KEY=/path/to/client-key.pem
NATS_TLS_CA=/path/to/ca.pem
NATS_USERNAME=example
NATS_PASSWORD=secret
```

These TLS variables are commented out in the generated `nats.env.example` file.
Uncomment them and provide the paths to your certificates to enable mTLS.
See [bus_template.md](bus_template.md) for instructions on creating test
certificates and running the Docker image.

The generated `subscriber.py` includes a simple `rate_limit` decorator. Apply
it to your message handler to throttle processing, e.g. `@rate_limit(10, 1)`
will allow up to ten messages per second.

The command copies the template to `src/deepthought/services/<name>` and
replaces `TemplateService` with a class named `<Name>Service`.

The generated directory contains `publisher.py`, `subscriber.py` and a
`Dockerfile` for containerized deployments. Customize these files to implement
your service logic.

Customize the generated files to implement your service logic.

## Multi-Agent Demo

The repository includes a demonstration of three lightweight agents exchanging messages using the `MemoryService` and a small HTTP LLM module. The agents are coordinated with a LangGraph state machine and live in `examples/multi_agent_demo.py`.

Start a local NATS server with `./scripts/start_nats.sh` and create the JetStream stream:

```bash
python -c "import asyncio, setup_jetstream; asyncio.run(setup_jetstream.setup_jetstream())"
```
The helper now retries connecting to NATS up to three times before giving up.

Install the optional dependency used by the demo:

```bash
pip install langgraph
```

Set `MODEL_PATH` to start the quantized edge model locally or set `LLM_ENDPOINT` to an existing `/generate` endpoint. Then launch the demo:

```bash
python examples/multi_agent_demo.py
```

You should see log lines similar to the following as the message circulates between agents:

```text
Agent 1 says: Hello from agent 1!
Agent 2 says: <generated reply>
Agent 3 says: <generated reply>
```

Ensure a NATS server is running on `localhost:4222` or set the `NATS_URL` environment variable accordingly.
