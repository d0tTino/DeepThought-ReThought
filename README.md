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
3.  **Knowledge Graph Memory:** `GraphMemory` currently uses the lightweight `networkx` library as an embedded graph store. See [`src/deepthought/modules/memory_graph.py`](src/deepthought/modules/memory_graph.py) for implementation details.
4.  **Adaptive Code Generation:** (Future Goal) Exploring techniques like template engines or JIT compilation (e.g., using AsmJit) for dynamic code optimization.
5.  **Neuromorphic Processing:** (Long-Term Research) Investigating brain-inspired computing principles via simulation (e.g., using Nengo).

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
        contains the packages needed for testing and linting.
4.  **Initialize JetStream:**
    *   Run the `setup_jetstream.py` script to create the necessary JetStream streams:
        ```bash
        python setup_jetstream.py
        ```
5.  **Run Components/Tests:** (Specific instructions TBD as components are developed)
6.  **CI-style setup:** A helper script mirrors the project\'s continuous integration workflow.
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

## Graph Memory Backend

The `GraphMemory` module uses the lightweight `networkx` library as an
embedded graph store. No external database service needs to be started.
Ensure the dependency is installed (included in `requirements.txt`).

## Social Graph Bot Example

An example Discord bot demonstrating social graph logging is available at `examples/social_graph_bot.py`. It records user interactions in a SQLite database, monitors channel activity, and forwards data to a Prism endpoint implemented in `examples/prism_server.py`.


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
3.  Run pytest with the project root added to `PYTHONPATH` so the `src` modules
    are discoverable:
    ```bash
    PYTHONPATH=src pytest
    ```
    Or, to run tests from a specific directory:
    ```bash
PYTHONPATH=src pytest tests/
```
4.  Check code style with flake8:
    ```bash
    flake8 src tests
    ```
    The default settings are configured in [.flake8](.flake8).

5.  Install and run the `pre-commit` hooks to automatically format and lint
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

6.  Alternatively, run the helper script that mirrors the CI workflow and
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
The workflow installs its dependencies from `requirements-ci.txt`.

You can run the same check locally:

```bash
python scripts/check_code_changes.py
```

The script compares your commit to the previous one and outputs `true` when any
non-comment code lines changed. The CI workflow uses this output to determine
whether tests and linters need to run.

## Contributing

This is currently a solo project developed by d0tTino. Contributions are not actively sought at this stage, but feel free to fork the repository or open issues for discussion.

---
