"""
# Perception Service

## Overview

The **PerceptionService** generates time-aligned embeddings for text, audio and video inputs.  Each modality is processed by a dedicated worker that emits embedding vectors with timestamps.  The service aligns these modality-specific streams onto a common grid, optionally fuses them into a single representation, and publishes the result.  The grid resolution defaults to the smallest hop across modalities but can be overridden with the `--grid-hop-size` CLI flag or the `DT_PERCEPTION_GRID_HOP_SIZE` environment variable.  Downstream modules can consume either the fused embeddings or per-modality outputs.

## Architecture and Flow

```mermaid
flowchart LR
    IR[dtr.input.received] --> T[Text Worker]
    IR --> A[Audio Worker]
    IR --> V[Video Worker]
    T --> F[Modality Fuser]
    A --> F
    V --> F
    F --> P[Publisher]
    P --> JS[(JetStream PERCEPTION stream)]
    P --> UE[(User Embeddings)]
    UE --> HM[Hierarchical Memory]
```

## JetStream Subjects

The service consumes perception inputs from:

- `dtr.input.received`
- `dtr.perception.extract`

After alignment and optional fusion, embeddings are published to:


- `dtr.perception.embeddings`

All messages are stored in the `PERCEPTION` stream created by `setup_jetstream.py`.
The stream now persists data to disk by default so replay jobs survive
process or node restarts.

## Event Schema

Embeddings are wrapped in a `PerceptionEmbeddingsEvent` containing encoder metadata, provenance, and a payload:

```json
{
  "event": "dtr.perception.embeddings",
  "version": 1,
  "encoders": [
    {"name": "TextPerceptionWorker"},
    {"name": "AudioPerceptionWorker"},
    {"name": "VideoPerceptionWorker"}
  ],
  "provenance": {"timestamp": 1713972000.0, "modalities": ["text", "audio", "video"]},
  "payload": {
    "message_id": "42",
    "user_id": "user",
    "fused": [[0.1, 0.2], [0.3, 0.4]],
    "by_modality": {
      "text": {
        "spans": [[0, 500], [500, 1000]],
        "embeddings": [[0.1, 0.2], [0.2, 0.3]],
        "encoders": [
          {"name": "TextPerceptionWorker"},
          {"name": "TextPerceptionWorker"}
        ]
      },
      "audio": {
        "spans": [[0, 500], [500, 1000]],
        "embeddings": [[0.3, 0.4], [0.4, 0.5]],
        "encoders": [
          {"name": "AudioPerceptionWorker"},
          {"name": "AudioPerceptionWorker"}
        ]
      },
      "video": {
        "spans": [[0, 500], [500, 1000]],
        "embeddings": [[0.5, 0.6], [0.6, 0.7]],
        "encoders": [
          {"name": "VideoPerceptionWorker"},
          {"name": "VideoPerceptionWorker"}
        ]
      }
    }
  }
}
```

Each `spans` entry captures `[start_ms, end_ms]` for the aligned hop, and `encoders` describe the model that produced the embedding.

### Extraction Requests

External tools can request fresh embeddings without going through the
`dtr.input.received` pipeline by publishing `PerceptionExtractEvent`
messages on `dtr.perception.extract`. The payload mirrors the arguments
accepted by `PerceptionService.run`:

```json
{
  "event": "dtr.perception.extract",
  "version": 1,
  "payload": {
    "message_id": "42",
    "user_id": "alice",
    "text": "optional raw text used to derive tokens",
    "text_tokens": [["Email", 0.0, 0.03], ["me", 0.03, 0.06]],
    "text_hop_size": 0.03,
    "audio_path": "/tmp/audio.wav",
    "video_path": null,
    "embeddings": [[0.1, 0.2, 0.3]],
    "spans": [[0, 30]],
    "modality_mask": {"text": [true]},
    "contribution_mask": {"text": [true]},
    "encoders": [{"name": "gte-small", "modality": "text"}],
    "provenance": {"timestamp": 1713972000.0},
    "audio_opt_in": true,
    "video_opt_in": false,
    "retain_media": false
  }
}
```

Only `message_id` and `user_id` are required; the service will derive hop
aligned tokens from `text` when `text_tokens` are omitted. When
`embeddings` is provided the existing fused vectors are republished,
which is useful for replay jobs. Otherwise the active perception workers
process any referenced media paths and emit new embeddings.

## Embedding Events

`PERCEPTION.EMBEDDINGS` events on the `dtr.perception.embeddings` subject carry the
vector representations produced for each message. The payload includes the
`message_id`, `user_id`, optional fused embeddings, and per-modality vectors with
their spans and encoder metadata. A simplified example:

```json
{
  "event": "dtr.perception.embeddings",
  "version": 1,
  "payload": {
    "message_id": "42",
    "user_id": "alice",
    "fused": [[0.1, 0.2, 0.3]],
    "by_modality": {
      "text": {
        "spans": [[0, 12]],
        "embeddings": [[0.4, 0.5, 0.6]],
        "encoders": [{"name": "gte-small"}]
      }
    }
  }
}
```

Durable consumers such as `memory-perception-consumer` can subscribe to the
stream and resume from the last acknowledged message after restarts. This makes
it easy for downstream modules to build analytics pipelines or persistent
indexes.

### Personalization

Storing embeddings per `user_id` enables simple personalization. A consumer can
aggregate a user's history and perform nearest-neighbour search to tailor model
responses or retrieve context relevant to that individual.

## Running the Service

Provide a NATS connection and optional model paths before launching:

```bash
export NATS_URL=nats://localhost:4222
# optional overrides for model names, cache directories or grid hop size
python -m deepthought.services.perception.cli --grid-hop-size 0.1 --listen
```

The listener consumes `dtr.input.received` messages and, unless disabled,
`dtr.perception.extract` requests before publishing aligned embeddings to
`dtr.perception.embeddings`. Use `--no-input-listener` or
`--no-extract-listener` to disable individual subscriptions and
`--extract-durable` to customize the durable consumer name for the
extraction channel.

## Durability Configuration

`setup_jetstream.py` configures the `PERCEPTION` stream with file-backed JetStream
storage and exposes several environment variables for tuning durability:

| Variable | Default | Description |
| --- | --- | --- |
| `PERCEPTION_RETENTION_POLICY` | `limits` | Retention policy (`limits`, `interest`, or `workqueue`). |
| `PERCEPTION_MAX_MSGS_PER_SUBJECT` | `10000` | Cap on messages per subject (set to `0` or unset for unlimited). |
| `PERCEPTION_MAX_MSGS` | unset | Global message limit across the stream. |
| `PERCEPTION_MAX_BYTES` | unset | Byte budget for the stream. |
| `PERCEPTION_MAX_AGE_SECONDS` | unset | Age limit before JetStream evicts entries. |

Unset values fall back to JetStream defaults (`-1`/unlimited). For production
commercialization we recommend budgeting disk space explicitly and pinning a
retention window, for example:

```bash
export PERCEPTION_RETENTION_POLICY=limits
export PERCEPTION_MAX_BYTES=$((80 * 1024 * 1024 * 1024))   # 80 GiB
export PERCEPTION_MAX_MSGS_PER_SUBJECT=0                   # unlimited per subject
export PERCEPTION_MAX_AGE_SECONDS=$((7 * 24 * 60 * 60))    # 7 days
```

These settings keep roughly a week of perception traffic while preventing the
stream from exhausting disk space. Increase `PERCEPTION_MAX_BYTES` or shorten
`PERCEPTION_MAX_AGE_SECONDS` based on observed throughput and retention
requirements.

## Replay and Monitoring

### Replaying Events

1. Provision the JetStream resources:
   ```bash
   python setup_jetstream.py
   ```
2. Inspect stored embeddings:
   ```bash
   nats consumer next PERCEPTION perception-replay --filter dtr.perception.embeddings --count=10 --json
   ```
3. To republish stored events with updated encoders or fusion settings:
   ```bash
   python scripts/replay_perception.py --nats-url nats://localhost:4222
   ```

   The replay script publishes `PerceptionExtractEvent` requests so the
   running perception service reprocesses each message and emits fresh
   `dtr.perception.embeddings` events.


### Monitoring with Weights & Biases

1. Install and log in to W&B:
   ```bash
   pip install wandb
   wandb login
   ```
2. Enable W&B in the perception service:
   ```bash
   export DT_WANDB_ENABLED=1
   export DT_WANDB_PROJECT=deepthought
   python -m deepthought.services.perception.cli --listen
   ```
3. Visit https://wandb.ai/ to view live metrics and uploaded artifacts.

### Training the Modality Fuser

Cached modality embeddings can be used to train the projection layer that
combines modalities. Save a `.npz` file containing a `target` array and one
array per modality (`text`, `audio`, `video`, etc.). A one-dimensional target is
automatically expanded to `(N, 1)`, and an optional `user_ids` array may be
included to preserve provenance for each sample. Train the fuser with:

```bash
perception-fuser-train --features path/to/data.npz --output fused.pt --epochs 3 \
  --batch-size 128 --dropout-prob 0.1 --device cuda:0 --seed 13 \
  --wandb-project my-project --wandb-entity my-team --wandb-group perception \
  --wandb-run-name fuser-v1
```

Key flags:

- `--dropout-prob` controls modality dropout during training.
- `--device` selects the torch device (`cpu`, `cuda:0`, etc.).
- `--seed` seeds Python, NumPy, and torch for reproducibility (shuffling can be
  disabled with `--no-shuffle`).
- `--wandb-*` options enable detailed Weights & Biases logging when `wandb` is
  installed. Per-epoch loss is emitted under the `train/loss` metric and the
  saved model is written to `--output`.

## Privacy and Consent

Perception inputs may contain personally identifiable information. Deployments **must** obtain user consent and disclose how media and derived embeddings are stored or shared.

- **Consent toggles:** set `PERCEPTION_REQUIRE_CONSENT=1` to ignore messages without an explicit `"consent": true` flag. Per-modality variables such as `DT_REQUIRE_AUDIO_CONSENT` and `DT_AUDIO_CONSENT` can enforce and grant consent for specific media types.
- **Data retention:** adjust the environment variables above to match your legal and operational policies. File-backed storage means embeddings persist until retention limits or quotas evict them.
- **External monitoring:** enabling W&B (`DT_WANDB_ENABLED=1`) sends metrics to a third-party service. Ensure this complies with your privacy policy.

Example configuration:

```bash
export PERCEPTION_REQUIRE_CONSENT=1
export DT_REQUIRE_AUDIO_CONSENT=1
export DT_AUDIO_CONSENT=1
export PERCEPTION_RETENTION_POLICY=workqueue
export DT_WANDB_ENABLED=1
export DT_WANDB_PROJECT=deepthought
```

These options allow deployments to honor user preferences and organizational data policies.
"""
