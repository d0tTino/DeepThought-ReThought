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
3.  **Knowledge Graph Memory:** (Planned) Implementation using an efficient open-source graph database (e.g., Memgraph, ArangoDB, Neo4j) to store and retrieve structured knowledge.
4.  **Adaptive Code Generation:** (Future Goal) Exploring techniques like template engines or JIT compilation (e.g., using AsmJit) for dynamic code optimization.
5.  **Neuromorphic Processing:** (Long-Term Research) Investigating brain-inspired computing principles via simulation (e.g., using Nengo).

## Current Status

* The foundational EDA using NATS/JetStream is implemented and tested with basic stub modules.
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
4.  **Initialize JetStream:**
    *   Run the `setup_jetstream.py` script to create the necessary JetStream streams:
        ```bash
        python setup_jetstream.py
        ```
5.  **Run Components/Tests:** (Specific instructions TBD as components are developed)

## Testing

Tests are implemented using the `pytest` framework. A running NATS server with
JetStream enabled is required. To run the tests:

1.  Ensure the NATS server is running and accessible.
2.  Install the Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run pytest:
    ```bash
    pytest
    ```
    Or, to run tests from a specific directory:
    ```bash
    pytest tests/
    ```

## Contributing

This is currently a solo project developed by d0tTino. Contributions are not actively sought at this stage, but feel free to fork the repository or open issues for discussion.

---
