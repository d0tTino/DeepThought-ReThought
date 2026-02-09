# Perception Event Subject Migration

DeepThought now publishes perception embedding events to both:

- the legacy fused subject: `dtr.perception.embeddings`
- modality-specific subjects:
  - `dtr.perception.image_embeddings`
  - `dtr.perception.audio_embeddings`
  - `dtr.perception.video_embeddings`

## Backward compatibility

Existing deployments that only subscribe to `dtr.perception.embeddings` continue to receive the full fused payload and do not require immediate changes.

## Correlation and confidence metadata

Perception extract and embedding payloads now carry correlation keys and confidence metadata:

- `input_id` and/or `message_id`
- `author_id`
- `channel_id`
- `confidence`
- `modality_confidence`

## Subscriber migration guidance

- Consumers that need a unified stream can keep using `dtr.perception.embeddings`.
- Consumers that only need one modality can subscribe to the corresponding modality subject to reduce downstream fan-in.
- During migration, subscribing to both fused and modality subjects is supported.
