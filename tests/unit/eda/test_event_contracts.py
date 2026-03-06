import pytest

from deepthought.eda.contracts import (
    CanonicalSubjects,
    EventEnvelope,
    decode_payload,
    decode_payload_or_envelope,
    to_canonical_subject,
    validate_envelope,
)
from deepthought.eda.events import (
    InputReceivedPayload,
    MemoryRetrievedPayload,
    PerceptionEmbeddingsPayload,
    PerceptionExtractPayload,
    ResponseCandidatesPayload,
    ResponseRankedPayload,
)


def test_legacy_subject_maps_to_canonical_and_payload_compatibility():
    payload = decode_payload("dtr.input.received", {"text": "hello"})
    assert (
        to_canonical_subject("dtr.input.received") == CanonicalSubjects.INPUT_RECEIVED
    )
    assert payload["user_input"] == "hello"


def test_event_envelope_roundtrip_validation():
    envelope = EventEnvelope.build(
        subject=CanonicalSubjects.RESPONSE_RANKED,
        payload={"final_response": "ok"},
        producer="selector",
    )
    validated = validate_envelope(envelope.__dict__)
    assert validated.subject == CanonicalSubjects.RESPONSE_RANKED


@pytest.mark.parametrize(
    ("payload_type", "raw"),
    [
        (
            InputReceivedPayload,
            {"user_input": "hi", "attachments": [{"url": "https://a", "size": 2}]},
        ),
        (MemoryRetrievedPayload, {"retrieved_knowledge": {"k": "v"}}),
        (ResponseCandidatesPayload, {"candidates": [{"text": "a", "confidence": 0.2, "source": "remote", "confidence_components": {"token_score": 0.4}, "safety_metadata": {"policy": "keyword_v1"}}]}),
        (
            ResponseRankedPayload,
            {"final_response": "a", "candidates": [{"text": "a", "confidence": 0.2, "source": "remote", "safety_metadata": {"policy": "keyword_v1"}}]},
        ),
        (
            PerceptionEmbeddingsPayload,
            {"message_id": "m1", "user_id": "u1", "by_modality": {}},
        ),
        (
            PerceptionExtractPayload,
            {"message_id": "m1", "user_id": "u1", "text_tokens": []},
        ),
    ],
)
def test_producer_consumer_schema_roundtrip(payload_type, raw):
    model = payload_type.from_dict(raw)
    encoded = model.to_json()
    decoded = payload_type.from_json(encoded)
    assert decoded == model


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("dtr.input.received", CanonicalSubjects.INPUT_RECEIVED),
        ("dtr.memory.retrieved", CanonicalSubjects.MEMORY_RETRIEVED),
        (
            "dtr.memory.retrieval.requested",
            CanonicalSubjects.MEMORY_RETRIEVAL_REQUESTED,
        ),
        ("dtr.social.signals.requested", CanonicalSubjects.SOCIAL_SIGNALS_REQUESTED),
        (
            "dtr.perception.interpret.requested",
            CanonicalSubjects.PERCEPTION_INTERPRET_REQUESTED,
        ),
        ("dtr.social.signals.retrieved", CanonicalSubjects.SOCIAL_SIGNALS_RETRIEVED),
        (
            "dtr.perception.interpret.retrieved",
            CanonicalSubjects.PERCEPTION_INTERPRET_RETRIEVED,
        ),
        ("dtr.context.assembled", CanonicalSubjects.CONTEXT_ASSEMBLED),
        ("dtr.response.candidates", CanonicalSubjects.RESPONSE_CANDIDATES),
        ("dtr.response.ranked", CanonicalSubjects.RESPONSE_RANKED),
        ("dtr.perception.embeddings", CanonicalSubjects.PERCEPTION_EMBEDDINGS),
        ("dtr.perception.extract", CanonicalSubjects.PERCEPTION_EXTRACT),
        ("dtr.social.perception", CanonicalSubjects.SOCIAL_PERCEPTION),
    ],
)
def test_legacy_subject_aliases_normalize_to_canonical_v1(legacy, canonical):
    assert to_canonical_subject(legacy) == canonical


def test_envelope_build_keeps_canonical_subject_stable():
    envelope = EventEnvelope.build(
        subject=CanonicalSubjects.INPUT_RECEIVED,
        payload={"user_input": "hi"},
        producer="discord_gateway",
    )
    assert envelope.subject == CanonicalSubjects.INPUT_RECEIVED


def test_invalid_input_payload_fails_safe():
    with pytest.raises(ValueError):
        InputReceivedPayload.from_dict(
            {"user_input": "x", "attachments": [{"url": "", "size": "1"}]}
        )


def test_invalid_envelope_fails_safe():
    with pytest.raises(ValueError):
        validate_envelope(
            {
                "schema_version": "1.0.0",
                "event_id": "bad-uuid",
                "trace_id": "also-bad",
                "causation_id": None,
                "created_at": "not-time",
                "producer": "svc",
                "subject": "dtr.input.received",
                "payload": {},
            }
        )


def test_decode_payload_or_envelope_accepts_legacy_payload():
    payload, meta = decode_payload_or_envelope(CanonicalSubjects.INPUT_RECEIVED, {"text": "hello"})
    assert payload["user_input"] == "hello"
    assert meta["trace_id"] is None


def test_decode_payload_or_envelope_accepts_enveloped_payload():
    envelope = EventEnvelope.build(
        subject=CanonicalSubjects.INPUT_RECEIVED,
        payload={"user_input": "hello"},
        producer="discord_gateway",
    )
    payload, meta = decode_payload_or_envelope(CanonicalSubjects.INPUT_RECEIVED, envelope.__dict__)
    assert payload["user_input"] == "hello"
    assert meta["trace_id"] == envelope.trace_id
    assert meta["producer"] == "discord_gateway"
