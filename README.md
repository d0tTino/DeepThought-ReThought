# DeepThought-ReThought

## Overview

DeepThought-ReThought is an experimental Artificial Intelligence (AI) project focused on exploring the boundaries of **computational efficiency** and **capability** under extreme resource constraints. Inspired by the "zero-budget" philosophy outlined in the "DeepThought Ultimate" research concept, this project aims to leverage open-source tools, innovative architectural patterns, and aggressive optimization techniques to build a powerful yet resource-frugal AI system.

The project is hosted on GitHub: [https://github.com/d0tTino/DeepThought-ReThought](https://github.com/d0tTino/DeepThought-ReThought)

## Goals

* **Extreme Efficiency:** Develop AI components that minimize computational cost (CPU, memory, energy) and financial expenditure.
* **Open Source Reliance:** Heavily utilize and contribute back to the open-source AI ecosystem (models, datasets, libraries, tools).
* **Modular Architecture:** Build a flexible system based on loosely coupled components communicating asynchronously.
* **Knowledge-Centric:** Incorporate structured knowledge representation (e.g., Knowledge Graphs) to ground reasoning and improve performance.
* **Push Boundaries:** Experiment with cutting-edge techniques in model optimization (QLoRA, quantization), adaptive code generation, and potentially neuromorphic principles (via simulation).

## Architecture (Current & Planned)

The system is built upon an **Event-Driven Architecture (EDA)** using NATS/JetStream for asynchronous communication between components.

Key planned/in-progress components include:

1.  **Core EDA Framework:** (Functional) Publishers, Subscribers, and Event definitions using NATS.
2.  **Optimized Language Model:** (In Progress) Utilizing small, open-source LLMs (< 3B parameters, e.g., Llama 3.2 3B Instruct) fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) techniques like QLoRA and further optimized with post-training quantization (e.g., AWQ). The `train_script.py` is provided for LLM fine-tuning. Running this script requires a suitable environment with a GPU and necessary CUDA libraries. It can be used to replicate the fine-tuning process described in `LLM_Fine_Tuning_Report.md`.
3.  **Hierarchical Memory Service:** combines `BasicMemory`, a planned Chroma-backed vector memory, and `KnowledgeGraphMemory` using Memgraph. These layers are orchestrated by the `MemoryService` to produce aggregated `MEMORY_RETRIEVED` events. See [docs/hierarchical_memory_service.md](docs/hierarchical_memory_service.md).
4.  **Reward Manager:** publishes user feedback as `RewardEvent` messages via JetStream, enabling future reinforcement or preference-based training. See [docs/reward_manager.md](docs/reward_manager.md).
5.  **Adaptive Code Generation:** (Future Goal) Exploring techniques like template engines or JIT compilation (e.g., using AsmJit) for dynamic code optimization.
6.  **Neuromorphic Processing:** (Long-Term Research) Investigating brain-inspired computing principles via simulation (e.g., using Nengo).

## Current Status

* The foundational EDA using NATS/JetStream is implemented and tested with the `BasicMemory` and `BasicLLM` reference modules.
* Environment setup for LLM fine-tuning (Roadmap Step 3) is complete (CUDA enabled).
* Currently proceeding with the LLM fine-tuning task: Loading `meta-llama/Llama-3.2-3B-Instruct` for QLoRA fine-tuning on the `databricks/databricks-dolly-15k` dataset.
*(Consider converting important information from the old .docx progress report to a Markdown file if still relevant).*

## Technology Stack

* **Primary Language:** Python 3.x
* **Messaging/EDA:** NATS, NATS JetStream (via `nats-py` library)
* **AI/ML:** PyTorch, Hugging Face Ecosystem (`transformers`, `peft`, `trl`, `datasets`, `accelerate`), `bitsandbytes`
* **Testing:** `pytest`, `pytest-asyncio`
* **Version Control:** Git, GitHub
* **(Planned) Database:** Open-source Graph Database (TBD)
* **(Optional) Unity Integration:** C# with the NATS C# Client

## Unity Integration (C#)

This repository includes C# scripts designed for NATS communication within a Unity game engine environment. These scripts facilitate the integration of Unity-based applications or simulations as components within the broader DeepThought-ReThought EDA.

Key scripts include:
*   `NatsService.cs`: Provides core connectivity and management for the NATS client within Unity.
*   `NatsJetStreamManager.cs`: Offers helper functionalities for setting up and managing NATS JetStream streams and consumers specifically tailored for Unity's asynchronous patterns (e.g., using UniTask or Coroutines).

Users would typically incorporate these scripts into their Unity projects and use them in conjunction with the official NATS C# client library.

## Setup & Usage

*(Note: This project is under active development. These are preliminary steps.)*

1.  **Prerequisites**:
    *   A running NATS server (latest stable version or X.Y.Z or later, with JetStream enabled using the `-js` flag) is required. You can download and run it from [nats.io](https://nats.io/) or use Docker.
    *   This repository no longer bundles a `nats-server` binary. Install your own NATS server instead of relying on an archived copy.
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/d0tTino/DeepThought-ReThought.git
    cd DeepThought-ReThought
    ```
3.  **Set up Python environment:**
    *   It's recommended to use a virtual environment (e.g., `venv`, `conda`).
    *   Install dependencies using the provided `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```
    *   A trimmed-down `requirements-ci.txt` is included for CI jobs and only
        contains the packages needed for testing and linting. The versions in
        this file are pinned so CI and local setups use exactly the same
        dependencies.
4.  **Initialize JetStream:**
    *   Run the `setup_jetstream.py` script to create the necessary JetStream streams:
        ```bash
        python setup_jetstream.py
        ```
5.  **Start Additional Services:**
    *   Launch Chroma for vector memory:
        ```bash
        docker run --rm -p 8000:8000 chromadb/chroma
        ```
    *   Launch Memgraph for knowledge graph storage:
        ```bash
        docker run --rm -p 7687:7687 memgraph/memgraph
        ```
    Set the GraphDAL environment variables as shown in [docs/hierarchical_memory_service.md](docs/hierarchical_memory_service.md).
6.  **Run Components/Tests:** (Specific instructions TBD as components are developed)
7.  **CI-style setup:** A helper script mirrors the project\'s continuous integration workflow.
    It installs dependencies, starts a temporary NATS server, initializes JetStream, and runs
    linters and tests only when code changes are detected:
    ```bash
    ./scripts/codex_setup.sh
    ```


### Configuration

BasicMemory stores user interaction history in a JSON file. Set the environment
variable `DT_MEMORY_FILE` to customize the location of this file:

```bash
export DT_MEMORY_FILE=/tmp/my_memory.json
```
If unset, the default `memory.json` in the current directory is used.
Settings can also be loaded from a configuration file by setting the `DT_CONFIG_FILE` environment variable.

Specify the NATS server address with the `NATS_URL` environment variable. If not set,
`nats://localhost:4222` is used by default:

```bash
export NATS_URL=nats://my-nats:4222
```

The optional offline search index used by `HierarchicalService` can be configured
via `DT_SEARCH_DB`:

```bash
export DT_SEARCH_DB=/data/wiki.db
```

`SchedulerService` runs periodic summarization and reminder tasks. Adjust the interval
between summaries with `DT_SCHEDULER_INTERVAL` (seconds):

```bash
export DT_SCHEDULER_INTERVAL=120
```

### Required environment variables

* `DISCORD_TOKEN` - Discord bot token
* `MONITOR_CHANNEL` - Channel ID the bot should monitor
* `NATS_URL` - Address of the NATS server

## Running a Local NATS Server

If you don't already have a NATS server running locally, you can start one easily.

* **Using Docker** (recommended):
  ```bash
  docker run --rm -p 4222:4222 -p 8222:8222 nats:latest -js
  ```

* **Using the `nats-server` binary**:
  ```bash
  nats-server -js
  ```

Once the server is up, create the required JetStream streams by running:
```bash
python setup_jetstream.py
```

This step is necessary before running the tests below.

## Recording Event Traces

Use `tools/record.py` to capture `INPUT_RECEIVED` and `RESPONSE_GENERATED` events:
```bash
python tools/record.py --output traces.jsonl
```
Press Ctrl+C to stop recording. Each line in the file is a JSON object with the event name and payload.


## Graph Memory Backend

The `GraphMemory` module uses the lightweight `networkx` library as an
embedded graph store. No external database service needs to be started.
Ensure the dependency is installed (included in `requirements.txt`).
If the graph file contains invalid JSON, `GraphMemory` will automatically
rewrite it with an empty graph so subsequent loads succeed.

### GraphDAL

`KnowledgeGraphMemory` now relies on **GraphDAL**, a thin data access
layer that persists nodes and edges in a Memgraph instance. Start
Memgraph with Docker and configure connection details with environment
variables:

```bash
docker run --rm -p 7687:7687 memgraph/memgraph
export MG_HOST=localhost
export MG_PORT=7687
export MG_USER=memgraph
export MG_PASSWORD=memgraph
```

See [docs/graphdal.md](docs/graphdal.md) for a minimal script that starts
the memory service and listens for `INPUT_RECEIVED` events.

## Social Graph Bot Example


An example Discord bot demonstrating social graph logging is available at
`examples/social_graph_bot.py`. It records user interactions in a SQLite
database, monitors channel activity, and forwards data to a Prism endpoint
implemented in `examples/prism_server.py`.

### Quick Start

Install the optional dependencies:

```bash
pip install discord.py aiohttp aiosqlite textblob vaderSentiment
```

If you choose the TextBlob backend, download its corpora:

```bash
python -m textblob.download_corpora
```

Set the environment variables used by the bot:

```bash
export DISCORD_TOKEN=your_token
export MONITOR_CHANNEL=1234567890
export SOCIAL_GRAPH_DB=/path/to/social_graph.db  # optional
export PRISM_ENDPOINT=http://localhost:5000/receive_data  # optional
export SENTIMENT_BACKEND=vader  # optional, defaults to textblob
```

`DISCORD_TOKEN` and `MONITOR_CHANNEL` are required. `NATS_URL` must also be set
to a valid NATS server address if the default `nats://localhost:4222` is not
appropriate. All other variables are optional.

Set `SENTIMENT_BACKEND` to either `textblob` or `vader` to choose the library
used for sentiment analysis. Any other value falls back to `textblob`.

Run the bot:

```bash
python examples/social_graph_bot.py
```

Alternatively, launch the same bot using the helper script at the project root:

```bash
python bot.py
```

### Running the Prism Server

To test Prism integration, start the simple Flask server:

```bash
python examples/prism_server.py
```

The bot's `send_to_prism` function posts JSON data to the endpoint
defined by the `PRISM_ENDPOINT` environment variable (default:
`http://localhost:5000/receive_data`).

### Idle Channel Monitoring

The example bot can gently revive a quiet channel. The `monitor_channels`
function checks for recent activity and posts a random prompt if no one has
spoken for a configurable number of minutes:

```python
async def monitor_channels(bot: discord.Client, channel_id: int) -> None:
    """Monitor a channel and occasionally speak during idle periods."""
    await bot.wait_until_ready()
    channel = bot.get_channel(channel_id)
    while not bot.is_closed():
        last_message = None
        async for msg in channel.history(limit=1):
            last_message = msg
            break

        respond_to = None
        send_prompt = False
        if not last_message:
            send_prompt = True
        else:
            idle_minutes = (
                discord.utils.utcnow() - last_message.created_at.replace(tzinfo=timezone.utc)
            ).total_seconds() / 60
            if idle_minutes >= IDLE_TIMEOUT_MINUTES:
                send_prompt = True
            elif BOT_CHAT_ENABLED:
                bots, humans = await who_is_active(channel)
                if bots and not humans:
                    age = await last_human_message_age(channel)
                    if age is None or age >= IDLE_TIMEOUT_MINUTES:
                        send_prompt = True
                        if last_message.author.bot:
                            respond_to = last_message

        if send_prompt:
            prompt = random.choice(idle_response_candidates)
            async with channel.typing():
                await asyncio.sleep(random.uniform(3, 10))
                if respond_to is not None:
                    await channel.send(prompt, reference=respond_to)
                else:
                    await channel.send(prompt)
        await asyncio.sleep(60)
```

Set the `IDLE_TIMEOUT_MINUTES` environment variable to control the inactivity
threshold. By default the bot waits five minutes before sending a prompt.
Enable bot-to-bot chatter by setting `BOT_CHAT_ENABLED=true`.

### Example Goals

The bot's `GoalScheduler` queues reminders formatted as `<seconds>:<message>`. They are forwarded to the `SchedulerService` in the background.

```python
bot.goal_scheduler.add_goal("60:Stretch your legs", priority=1)
bot.goal_scheduler.add_goal("300:Time for a break", priority=2)
```


## Discord Bot Roadmap

For a detailed overview of the Discord bot progress, see [docs/discord_bot_roadmap.md](docs/discord_bot_roadmap.md).
For the comprehensive plan outlining each phase, see [docs/discord_bot_phase_report.md](docs/discord_bot_phase_report.md).



## Basic Modules

Two lightweight reference modules show how components can interact through NATS:

* **BasicMemory** -- subscribes to `INPUT_RECEIVED` events, stores each user input in a local `memory.json` file and then publishes a `MEMORY_RETRIEVED` event containing the most recent entries.
* **BasicLLM** -- listens for `MEMORY_RETRIEVED`, runs a small language model to generate a reply, and publishes `RESPONSE_GENERATED`. This module requires the optional heavy dependencies `transformers` and `torch`.

### Example Workflow

1. The `InputHandler` emits an `INPUT_RECEIVED` event when it receives a message.
2. `BasicMemory` logs the text to `memory.json` and publishes a `MEMORY_RETRIEVED` event with the last few inputs.
3. `BasicLLM` generates a response from those facts and publishes a `RESPONSE_GENERATED` event.
4. The `OutputHandler` (or another consumer) can then deliver the response to the user.


## Testing

Tests are implemented using the `pytest` framework. To run the tests:

1.  Ensure a NATS server with JetStream enabled is running and accessible (see [Running a Local NATS Server](#running-a-local-nats-server)).
    * You can start one quickly by running:
      ```bash
      ./scripts/start_nats.sh
      ```
    * If no server is available, tests that require NATS will be skipped automatically.
    * **For full test coverage, JetStream must be available.** Tests that rely on persistence or streaming features will be skipped when JetStream is not running.
2.  Navigate to the root directory of the project.
3.  Install the dependencies used for testing:
    ```bash
    pip install -r requirements-ci.txt
    ```
    The pinned versions in this file match the CI environment.
    (Use `requirements.txt` instead if you plan to run the full application.)
4.  Run pytest with the project root added to `PYTHONPATH` so the `src` modules
    are discoverable:
    ```bash
    PYTHONPATH=src pytest
    ```
    Or, to run tests from a specific directory:
    ```bash
PYTHONPATH=src pytest tests/
```
5.  Check code style with flake8:
    ```bash
    flake8 src tests
    ```
    The default settings are configured in [.flake8](.flake8).

6.  Install and run the `pre-commit` hooks to automatically format and lint
    your changes:
    ```bash
    pip install pre-commit
    pre-commit install
    ```
    To run all hooks manually:
    ```bash
    pre-commit run --all-files
    ```
    The configuration is provided in [.pre-commit-config.yaml](.pre-commit-config.yaml).

7.  Alternatively, run the helper script that mirrors the CI workflow and
    automates setup:
    ```bash
    ./scripts/codex_setup.sh
    ```
    This script installs dependencies, starts a local NATS server, sets up
    JetStream, and executes flake8 and pytest **only when code has changed**.

## Continuous Integration

The project includes a GitHub Actions workflow that runs `flake8` and
the test suite whenever code changes are detected. Detection is handled by
[`scripts/check_code_changes.py`](scripts/check_code_changes.py), which
inspects diffs and skips CI when only documentation or comments change.
The workflow installs its dependencies from `requirements-ci.txt` and
uses the pinned versions to ensure consistent results across runs. It
only triggers for changes on the `main` and `develop` branches. Each run
is isolated with a concurrency group so outdated jobs are cancelled
automatically. If you would like to run the workflow on your own machine,
see [docs/ci.md](docs/ci.md) for instructions on registering a self-hosted runner.

You can run the same check locally:

```bash
python scripts/check_code_changes.py
```

The script compares your commit to the previous one and outputs `true` when any
non-comment code lines changed. The CI workflow uses this output to determine
whether tests and linters need to run.

## Manual Release

The `Release` workflow runs automatically whenever a GitHub Release is
published. You can also trigger it manually thanks to the
`workflow_dispatch` event. Navigate to the **Actions** tab on GitHub,
select the **Release** workflow, click **Run workflow**, and confirm to
start a manual release. This is useful for verifying the packaging steps
without creating a new tag.

## Contributing

This is currently a solo project developed by d0tTino. Contributions are not actively sought at this stage, but feel free to fork the repository or open issues for discussion.

---
