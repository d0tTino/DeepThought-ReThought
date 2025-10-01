"""Round-trip tests for :mod:`deepthought.eda.events` perception payloads."""

from deepthought.eda.events import (
    EncoderMetadata,
    ModalityEmbeddings,
    PerceptionEmbeddingsEvent,
    PerceptionEmbeddingsPayload,
)


def _encoder(name: str, modality: str, dim: int) -> EncoderMetadata:
    return EncoderMetadata(name=name, modality=modality, dim=dim)


def test_roundtrip_without_fused_vectors():
    payload = PerceptionEmbeddingsPayload(
        message_id="msg-no-fused",
        user_id="user-1",
        spans=[[0, 5]],
        modality_mask={"text": [True]},
        contribution_mask={"text": [False]},
        by_modality={
            "text": ModalityEmbeddings(
                spans=[[0, 5]],
                embeddings=[[0.1, 0.2]],
                encoders=[_encoder("text-enc", "text", 2)],
                mask=[True],
            )
        },
    )

    event = PerceptionEmbeddingsEvent(
        encoders=[_encoder("text-enc", "text", 2)],
        provenance={"source": "unit-test"},
        payload=payload,
    )

    encoded = event.to_json()
    decoded = PerceptionEmbeddingsEvent.from_json(encoded)

    assert decoded.payload == payload
    assert decoded.payload.fused is None
    assert decoded.payload.contribution_mask == payload.contribution_mask
    assert decoded.encoders == event.encoders


def test_roundtrip_with_multiple_modalities_and_masks():
    payload = PerceptionEmbeddingsPayload(
        message_id="msg-multi",
        user_id="user-2",
        fused=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        spans=[[0, 10], [10, 20]],
        modality_mask={
            "audio": [True, False],
            "vision": [True, True],
        },
        contribution_mask={
            "audio": [True, False],
            "vision": [False, True],
        },
        by_modality={
            "audio": ModalityEmbeddings(
                spans=[[0, 10], [10, 20]],
                embeddings=[[0.05, 0.07], [0.08, 0.09]],
                encoders=[_encoder("audio-enc", "audio", 2)],
                mask=[True, False],
            ),
            "vision": ModalityEmbeddings(
                spans=[[0, 10], [10, 20]],
                embeddings=[[0.11, 0.12], [0.13, 0.14]],
                encoders=[_encoder("vision-enc", "vision", 2)],
                mask=[True, True],
            ),
        },
    )

    event = PerceptionEmbeddingsEvent(
        encoders=[
            _encoder("audio-enc", "audio", 2),
            _encoder("vision-enc", "vision", 2),
        ],
        provenance={"source": "unit-test"},
        payload=payload,
    )

    encoded = event.to_json()
    decoded = PerceptionEmbeddingsEvent.from_json(encoded)

    assert decoded.payload == payload
    assert decoded.payload.fused == payload.fused
    assert decoded.payload.modality_mask == payload.modality_mask
    assert decoded.payload.contribution_mask == payload.contribution_mask
    assert set(decoded.payload.by_modality) == {"audio", "vision"}
    assert decoded.payload.by_modality["audio"].mask == [True, False]
    assert decoded.payload.by_modality["vision"].mask == [True, True]
    assert decoded.encoders == event.encoders
