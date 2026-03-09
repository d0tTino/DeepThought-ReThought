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

export DISCORD_BOT_TOKEN=your_token
# DISCORD_TOKEN remains a deprecated legacy alias.
export MONITOR_CHANNEL=1234567890
export SOCIAL_GRAPH_DB=/path/to/social_graph.db  # optional
export PRISM_ENDPOINT=http://localhost:5000/receive_data  # optional

python examples/social_graph_bot.py
```

The `examples/social_graph_bot.py` script logs user interactions in a SQLite database, monitors channel activity, and forwards data to a Prism endpoint implemented in `examples/prism_server.py`. The FastAPI server expects an `Authorization` header containing one of the tokens specified in the `PRISM_TOKENS` environment variable.

The knowledge graph memory uses GraphDAL to persist data in Memgraph. Start Memgraph with `docker run --rm -p 7687:7687 memgraph/memgraph` and see [docs/graphdal.md](graphdal.md) for a full example service.

### Goal Scheduler

Run the goal scheduler in the background to publish queued intentions automatically:

```python
from deepthought.goal_scheduler import GoalScheduler

sched = GoalScheduler()
sched.start(publisher, interval=5.0)  # seconds
# ... queue goals with sched.add_goal or sched.queue_intention ...
await sched.stop()
```

Projects board reminders now enqueue a single message that already points back to
the project thread when one exists. If the downstream scheduler service is
unavailable, the bot immediately publishes the reminder payload instead of
dropping it so project discussions still receive timely nudges. When a thread
mention is included, the fallback path also attempts to post the reminder
directly into that thread so Discord users still see the update in context.

Holiday tagging follows the same flow: when `PROJECT_HOLIDAY_LOCALE` (or its
`PROJECTS_HOLIDAY_LOCALE` alias) is set, the board looks up regional holidays
for due dates, toggles the 🎁 tag automatically, and exposes a dashboard button
that filters the Kanban columns down to only those seasonal projects.

## Additional Safety Features

Recent iterations introduced several controls to moderate conversations:

* **Manipulative language detection** – incoming messages are scored by `manipulation_score`. Higher scores lower the sender's trust value and are logged in the thoughts channel.
* **Emotion detection** – messages are analyzed for basic emotions to inform contextual replies.
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

### THOUGHT_CHANNEL usage and permissions

Set the variable to the numeric ID of a Discord channel that only the bot and
trusted maintainers can access.

* Grant the bot permission to send messages.
* Deny read access to regular members to keep internal assessments hidden.
* Avoid referencing thought logs in public channels.

## Manipulation Detection Flow

Set ``MANIP_MODEL_PATH`` to load a Hugging Face classifier used to categorize
manipulative messages. If unset, the bot falls back to phrase heuristics.

```bash
pip install transformers
export MANIP_MODEL_PATH=/path/to/manip-model
```

Messages flagged as manipulative lower the sender's trust and are logged in the
thought channel.

## Dynamic Deception Replies

Deceptive answers can now be generated dynamically. Enable the feature with:

```bash
export ALLOW_DECEPTION=true
export DECEPTION_REPLY_MODE=dynamic
```

The first probing question triggers a short text-generation run. The fabricated
reply is stored so identical questions return the same response later.

## Bidirectional Relationship Tracking

Interactions are logged for both ``(user, target)`` and ``(target, user)``. The
``PersonaManager`` selects a persona based on this mutual affinity score.
Specify a database path if you need persistence:

```bash
export SOCIAL_GRAPH_DB=/path/to/social_graph.db
```
