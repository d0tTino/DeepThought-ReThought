from deepthought.eda.events import (
    EncoderMetadata,
    ModalityEmbeddings,
    PerceptionEmbeddingsEvent,
    PerceptionEmbeddingsPayload,
    InputReceivedPayload,
)


def test_perception_embeddings_event_from_json_roundtrip():
    payload = PerceptionEmbeddingsPayload(
        message_id="msg-123",
        user_id="user-456",
        fused=[[0.1, 0.2], [0.3, 0.4]],
        spans=[[0, 10], [10, 20]],
        modality_mask={"audio": [True, False]},
        contribution_mask={"audio": [True, True]},
        by_modality={
            "audio": ModalityEmbeddings(
                spans=[[0, 10], [10, 20]],
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                encoders=[EncoderMetadata(name="enc", modality="audio", dim=2)],
                mask=[True, False],
            )
        },
    )

    event = PerceptionEmbeddingsEvent(
        encoders=[EncoderMetadata(name="enc", modality="audio", dim=2)],
        provenance={"source": "unit-test"},
        payload=payload,
    )

    encoded = event.to_json()
    decoded = PerceptionEmbeddingsEvent.from_json(encoded)

    assert decoded.payload == payload
    assert decoded.payload.contribution_mask == payload.contribution_mask
    assert decoded.payload.modality_mask == payload.modality_mask
    assert decoded.encoders == event.encoders
    assert decoded.provenance == event.provenance


def test_input_received_payload_ignores_malformed_attachments():
    payload = InputReceivedPayload.from_dict(
        {
            "user_input": "hello",
            "attachments": [
                {"url": "https://x.test/a.png", "content_type": "image/png", "filename": "a.png", "size": 7},
                {"url": "", "content_type": "image/png"},
                {"url": "https://x.test/b.mp3", "size": "bad"},
                "invalid",
            ],
        }
    )

    assert payload.user_input == "hello"
    assert payload.attachments is not None
    assert len(payload.attachments) == 2
    assert payload.attachments[0].size == 7
    assert payload.attachments[1].size is None
