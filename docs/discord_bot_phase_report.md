# Discord Bot Phase Report

This document consolidates the Phase 1 and Phase 2 progress notes for the Discord bot integration. It also reproduces setup commands and example code snippets from the repository for quick reference.

## Phase 1: Foundation

- Establish event-driven architecture using NATS and JetStream.
- Implement a minimal Discord bot that logs user interactions.
- Create initial tests for message flow and JetStream integration.
- Document environment setup and hardware considerations.

### Setup Steps

1. Clone the repository and install dependencies.
2. Start a local NATS server with JetStream enabled:

```bash
# Using Docker
docker run --rm -p 4222:4222 -p 8222:8222 nats:latest -js

# Or using the binary
nats-server -js
```

3. Initialize the required JetStream streams:

```bash
python setup_jetstream.py
```

## Phase 2: Social Graph Logging

- Extend the bot with a social graph that records interactions.
- Add SQLite storage and basic sentiment analysis.
- Provide a Prism server example for forwarding data for analysis.
- Plan for knowledge graph integration and advanced response generation.

### Quick Start Example

Install optional packages and run the bot:

```bash
pip install discord.py aiohttp aiosqlite textblob

export DISCORD_TOKEN=your_token
export MONITOR_CHANNEL=1234567890

python examples/social_graph_bot.py
```

The `examples/social_graph_bot.py` script logs user interactions in a SQLite database, monitors channel activity, and forwards data to a Prism endpoint implemented in `examples/prism_server.py`.
