# Orchestrator Quick Start

The orchestrator starts multiple services with a single NATS connection. Each service is registered via an entry point in the `deepthought.services` group.

Create a configuration file listing the services to launch:

```yaml
services:
  - demo
  - codegen
```

Start a local NATS server with JetStream enabled then run:

```bash
dtrt orchestrate orchestrator.yaml
```

Publishing an `INPUT_RECEIVED` event will now flow through the `DemoService` and the `CodeGenerationService`, demonstrating end-to-end event handling.
