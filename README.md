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
2.  **Optimized Language Model:** (In Progress) Utilizing small, open-source LLMs (< 3B parameters, e.g., Llama 3.2 3B Instruct) fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) techniques like QLoRA and further optimized with post-training quantization (e.g., AWQ).
3.  **Knowledge Graph Memory:** (Planned) Implementation using an efficient open-source graph database (e.g., Memgraph, ArangoDB, Neo4j) to store and retrieve structured knowledge.
4.  **Adaptive Code Generation:** (Future Goal) Exploring techniques like template engines or JIT compilation (e.g., using AsmJit) for dynamic code optimization.
5.  **Neuromorphic Processing:** (Long-Term Research) Investigating brain-inspired computing principles via simulation (e.g., using Nengo).

## Current Status

* The foundational EDA using NATS/JetStream is implemented and tested with basic stub modules.
* Environment setup for LLM fine-tuning (Roadmap Step 3) is complete (CUDA enabled).
* Currently proceeding with the LLM fine-tuning task: Loading `meta-llama/Llama-3.2-3B-Instruct` for QLoRA fine-tuning on the `databricks/databricks-dolly-15k` dataset.

*(See `DeepThought Progress vs. Roadmap Analysis (2025-04-20).docx` for detailed status against the project roadmap).*

## Technology Stack

* **Primary Language:** Python 3.x
* **Messaging/EDA:** NATS, NATS JetStream (via `nats-py` library)
* **AI/ML:** PyTorch, Hugging Face Ecosystem (`transformers`, `peft`, `trl`, `datasets`, `accelerate`), `bitsandbytes`
* **Testing:** `pytest`, `pytest-asyncio`
* **Version Control:** Git, GitHub
* **(Planned) Database:** Open-source Graph Database (TBD)

## Setup & Usage

*(Note: This project is under active development. These are preliminary steps.)*

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/d0tTino/DeepThought-ReThought.git](https://github.com/d0tTino/DeepThought-ReThought.git)
    cd DeepThought-ReThought
    ```
2.  **Set up Python environment:**
    * It's recommended to use a virtual environment (e.g., `venv`, `conda`).
    * Install dependencies (requirements file TBD, see recent steps for current packages):
        ```bash
        # Example based on current needs - formal requirements.txt pending
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
        pip install transformers peft trl datasets accelerate bitsandbytes nats-py pytest pytest-asyncio
        ```
3.  **Run NATS Server:**
    * A NATS server instance needs to be running for the EDA components to communicate. Download and run from [nats.io](https://nats.io/) or use Docker.
4.  **Run Components/Tests:** (Specific instructions TBD as components are developed)
    * Run integration tests using `pytest`.

## Contributing

This is currently a solo project developed by d0tTino. Contributions are not actively sought at this stage, but feel free to fork the repository or open issues for discussion.

---
