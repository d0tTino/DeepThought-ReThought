# Evaluation Framework

This document describes how to replay Discord interactions through NATS and compare
responses against a golden dataset.

1. Capture a raw trace using `tools/record.py`.
2. Prepare a YAML file containing the expected replies and qualitative ratings.
3. Run `tools/discord_replay.py` with the `--golden` option:
   ```bash
   python tools/discord_replay.py my_trace.jsonl \
       --golden tests/interaction_samples/greeting.yaml \
       --output replay.jsonl --metrics metrics.json
   ```
4. The script collects new bot replies, computes BLEU and ROUGE-L scores against
   the golden responses, and writes average latency and throughput metrics.
   If ratings are provided in the YAML file, the average rating is reported as
   `avg_rating` in the metrics JSON.
