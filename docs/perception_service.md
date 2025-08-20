# Perception Service

## Overview

The **PerceptionService** scores text, image, and audio inputs for social cues such as flirtation, avoidance, manipulation, sarcasm and supportiveness. Each modality is processed separately and a simple fusion step averages the signals into a combined vector. Downstream modules can use the fused or per-modality scores to adjust trust levels, choose personas or trigger safeguards.

## JetStream Subjects

The service listens for incoming messages on:

- `dtr.input.text`
- `dtr.input.image`
- `dtr.input.audio`

After scoring, results are published to JetStream under:

- `dtr.perception.text`
- `dtr.perception.image`
- `dtr.perception.audio`
- `dtr.perception.fused`

All subjects are persisted in the `PERCEPTION` stream created by `setup_jetstream.py`.

## Event Schema

A fused perception event bundles the modality scores and the averaged result:

```json
{
  "input_id": "42",
  "text": {"manipulation": 0.1},
  "image": {"manipulation": 0.0},
  "audio": {"manipulation": 0.2},
  "fused": {
    "flirtation": 0.2,
    "avoidance": 0.1,
    "manipulation": 0.1,
    "sarcasm": 0.3,
    "supportiveness": 0.4
  },
  "timestamp": "2024-05-01T12:00:00Z"
}
```

Per-modality events omit the other sections and include only the scores for that input type.

## Running the Service

Provide a NATS connection and optional model paths before launching:

```bash
export NATS_URL=nats://localhost:4222
# optional custom model paths
export SOCIAL_PERCEPTION_MODEL=$(pwd)/models/social_perception      # text
export PERCEPTION_IMAGE_MODEL=$(pwd)/models/image_perception        # image
export PERCEPTION_AUDIO_MODEL=$(pwd)/models/audio_perception        # audio
python -m deepthought.services.perception_service
```

The module continuously publishes scored events to the subjects listed above.

## Replaying Perception Events

`setup_jetstream.py` provisions a `PERCEPTION` stream with durable consumers `memory-perception-consumer` and `analytics-perception-consumer`. Use the NATS CLI to inspect past events. For example, replay the next ten fused events:

```bash
nats consumer next PERCEPTION memory-perception-consumer --filter dtr.perception.fused --count=10 --json
```

Swap the `--filter` value for `dtr.perception.text`, `dtr.perception.image`, or `dtr.perception.audio` to retrieve scores for individual modalities.

## Privacy Controls

The perception service provides basic controls over user consent and data retention:

- **Consent toggle:** set `PERCEPTION_REQUIRE_CONSENT=1` to require incoming messages to include a `"consent": true` flag. Events without consent are ignored.
- **Retention policy:** the `PERCEPTION` stream defaults to the JetStream `limits` retention policy. Override `PERCEPTION_RETENTION_POLICY` with `limits`, `interest`, or `workqueue` when provisioning streams to control how long scored events are stored.

Example configuration:

```bash
export PERCEPTION_REQUIRE_CONSENT=1
export PERCEPTION_RETENTION_POLICY=workqueue
```

These options allow deployments to honor user preferences and organizational data policies.
