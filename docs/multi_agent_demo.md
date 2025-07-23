# Multi-Agent Demo

This guide shows how to run `examples/multi_agent_demo.py` using a lightweight edge model container.

## Prerequisites

1. **Start NATS**
   
   Launch a local NATS server with JetStream enabled:
   
   ```bash
   ./scripts/start_nats.sh
   ```
   
   Then create the required stream:
   
   ```bash
   python -c "import asyncio, setup_jetstream; asyncio.run(setup_jetstream.setup_jetstream())"
   ```
2. **Install Dependencies**
   
   Install the main requirements and the optional LangGraph dependency:
   
   ```bash
   pip install -r requirements.txt
   pip install langgraph
   ```
3. **Model Access**
   
   Either set `MODEL_PATH` to a local quantized model or provide `LLM_ENDPOINT` pointing to a `/generate` HTTP endpoint.

## Build the Edge Image

Use the provided Dockerfile to build the lightweight inference image:

```bash
docker build -f docker/Dockerfile.edge -t dtrt-edge .
```

To select a different base model, pass the `MODEL_NAME` build argument.

## Launch with Metrics Enabled

Set the `EDGE_IMAGE` environment variable so the demo starts the container automatically. Specify `METRICS_PORT` to expose Prometheus metrics:

```bash
EDGE_IMAGE=dtrt-edge METRICS_PORT=8000 python examples/multi_agent_demo.py
```

The script terminates the container when finished. Metrics will be available on the specified port.
