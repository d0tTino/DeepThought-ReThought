# Development Tasks

This document lists actionable tasks derived from the project documentation and reports. Each item is short and testable so contributors can track progress.

## LLM Fine-Tuning
- [x] Set up a CUDA-enabled environment and install dependencies from `requirements.txt`.
- [x] Run `train_script.py` to fine-tune `meta-llama/Llama-3.2-3B-Instruct` on `databricks/databricks-dolly-15k` using QLoRA.
- [x] Monitor training metrics (loss, evaluation metrics) and GPU utilization.
- [x] Evaluate the resulting adapter on a held-out test set and save metrics.

## NATS Setup
- [x] Start a local NATS server with JetStream enabled.
- [x] Execute `setup_jetstream.py` to create required streams before running modules or tests.

## Social Graph Bot
- [x] Configure Discord intents so the bot can read message content.
- [x] Implement situational awareness checks (idle monitoring, bot chatter limits) in `examples/social_graph_bot.py`.
- [x] Forward interaction data to the Prism endpoint in `examples/prism_server.py`.
- [ ] Implement bullying detection and optional snide replies in `examples/social_graph_bot.py`.
- [ ] Track per-user/channel sentiment trends for theme modeling.
- [ ] Respect `do_not_mock` flags for sensitive users.
- [x] Add tests verifying the new features.

## Documentation
- [x] Convert information from the old `.docx` progress report to Markdown.
- [x] Record any environment-specific steps in `docs/setup_log.md`.

Remaining tasks include implementing bullying detection, sentiment trend tracking, and honoring `do_not_mock` flags.

For additional context on the Discord bot development phases, see
[Discord Bot Phase Report](discord_bot_phase_report.md).

Additional planning notes and a more detailed checklist are available in
[Prism Discord Bot Report](prism_discord_bot_report.md) and
[Prism Discord Bot Tasks](prism_discord_bot_tasks.md).

