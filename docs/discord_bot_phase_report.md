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
python -c "import asyncio, setup_jetstream; asyncio.run(setup_jetstream.setup_jetstream())"
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
```

After installing TextBlob, download its corpora:

```bash
python -m textblob.download_corpora

export DISCORD_TOKEN=your_token
export MONITOR_CHANNEL=1234567890
export SOCIAL_GRAPH_DB=/path/to/social_graph.db  # optional
export PRISM_ENDPOINT=http://localhost:5000/receive_data  # optional

python examples/social_graph_bot.py
```

The `examples/social_graph_bot.py` script logs user interactions in a SQLite database, monitors channel activity, and forwards data to a Prism endpoint implemented in `examples/prism_server.py`. The FastAPI server expects an `Authorization` header containing one of the tokens specified in the `PRISM_TOKENS` environment variable.

The knowledge graph memory uses GraphDAL to persist data in Memgraph. Start Memgraph with `docker run --rm -p 7687:7687 memgraph/memgraph` and see [docs/graphdal.md](graphdal.md) for a full example service.

## Additional Safety Features

Recent iterations introduced several controls to moderate conversations:

* **Manipulative language detection** – incoming messages are scored by `manipulation_score`. Higher scores lower the sender's trust value and are logged in the thoughts channel.
* **Rate limiting** – replies are throttled with `UserRateLimiter`. Tune the delay via the `USER_REPLY_RATE_SECONDS` environment variable.
* **Deception memory** – when `ALLOW_DECEPTION=true`, the bot answers probing questions with a predefined message. Each deceptive reply is stored for future reference.
* **Bot-to-bot cooldown** – to prevent chatter loops, set `PLAYFUL_REPLY_TIMEOUT_MINUTES` and cap active bots with `MAX_BOT_SPEAKERS`.

## Local Social Perception and Thought Logging

Download the social perception classifier and point `SOCIAL_PERCEPTION_MODEL` to the local weights:

```bash
pip install huggingface_hub
huggingface-cli download myorg/social-cue-classifier --local-dir ./models/social_perception
export SOCIAL_PERCEPTION_MODEL=$(pwd)/models/social_perception
```

Enable thought logging by specifying a channel ID:

```bash
export THOUGHT_CHANNEL=9876543210
```

When configured, the bot posts manipulation scores and other notes to this private channel.
