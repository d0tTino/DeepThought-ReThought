# Development Tasks

This document lists actionable tasks derived from the project documentation and reports. Each item is short and testable so contributors can track progress.

## LLM Fine-Tuning
- [ ] Set up a CUDA-enabled environment and install dependencies from `requirements.txt`.
- [ ] Run `train_script.py` to fine-tune `meta-llama/Llama-3.2-3B-Instruct` on `databricks/databricks-dolly-15k` using QLoRA.
- [ ] Monitor training metrics (loss, evaluation metrics) and GPU utilization.
- [ ] Evaluate the resulting adapter on a held-out test set and save metrics.

## NATS Setup
- [ ] Start a local NATS server with JetStream enabled.
- [ ] Execute `setup_jetstream.py` to create required streams before running modules or tests.

## Social Graph Bot
- [ ] Configure Discord intents so the bot can read message content.
- [ ] Implement situational awareness checks (idle monitoring, bot chatter limits) in `examples/social_graph_bot.py`.
- [ ] Forward interaction data to the Prism endpoint in `examples/prism_server.py`.

## Documentation
- [ ] Convert information from the old `.docx` progress report to Markdown.
- [ ] Record any environment-specific steps in `docs/setup_log.md`.

