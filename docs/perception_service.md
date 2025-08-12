# Perception Service

The **PerceptionService** evaluates incoming user messages for social cues
like flirtation, avoidance, manipulation, sarcasm and supportiveness.
Downstream modules can use the scores to adjust trust levels, choose
personas or trigger safeguards.

## Event Schema

PerceptionService subscribes to `dtr.input.received` events and publishes
`dtr.perception.scored` once analysis completes. A published payload
contains the original `input_id` plus the classifier scores:

```json
{
  "input_id": "42",
  "perception": {
    "flirtation": 0.2,
    "avoidance": 0.1,
    "manipulation": 0.0,
    "sarcasm": 0.3,
    "supportiveness": 0.4
  },
  "timestamp": "2024-05-01T12:00:00Z"
}
```

Consumers can store these events in memory, update trust scores or trigger
policy checks.

## CLI Usage

The service uses the social perception model specified by
`SOCIAL_PERCEPTION_MODEL` or the bundled defaults. Start the service after
configuring a NATS connection:

```bash
export NATS_URL=nats://localhost:4222
# optional custom model path
export SOCIAL_PERCEPTION_MODEL=$(pwd)/models/social_perception
python -m deepthought.services.perception_service
```

The module listens for new inputs and continuously publishes
`dtr.perception.scored` events to the bus.

