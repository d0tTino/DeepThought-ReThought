# Evaluation Framework

This document describes how to replay Discord interactions through NATS and compare
responses against a golden dataset.

1. Capture a raw trace using `tools/record.py`.
2. Prepare a YAML file containing the expected replies and qualitative ratings.
   Golden conversations shipped with the repository live under
   `tests/interaction_samples/golden/` and can be referenced by name.
3. Run `tools/discord_replay.py` with the `--golden` option:
   ```bash
   python tools/discord_replay.py my_trace.jsonl \
       --golden qa \
       --output replay.jsonl --metrics metrics.json
   ```
4. The script collects new bot replies, computes BLEU and ROUGE-L scores against
   the golden responses, and writes average latency and throughput metrics.
   Persona state and affinity for each turn are stored in the replay JSONL file.
   If ratings are provided in the YAML file, the average rating is reported as
   `avg_rating` in the metrics JSON.

## Contributing additional scenarios
To expand the golden dataset, place YAML files under `tests/interaction_samples/golden/`.
Each file should contain a list of interaction objects with `input`, `expected`, and `rating` fields.
Ratings should be integers from 1 to 5.
Use the existing samples as a reference.
