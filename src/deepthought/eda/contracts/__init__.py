"""Canonical EDA event contracts and migration compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID, uuid4


class CanonicalSubjects:
    """Canonical, versioned subject names used by production traffic."""

    INPUT_RECEIVED = "dtr.input.received.v1"
    MEMORY_RETRIEVED = "dtr.memory.retrieved.v1"
    MEMORY_RETRIEVAL_REQUESTED = "dtr.memory.retrieval.requested.v1"
    SOCIAL_SIGNALS_REQUESTED = "dtr.social.signals.requested.v1"
    PERCEPTION_INTERPRET_REQUESTED = "dtr.perception.interpret.requested.v1"
    SOCIAL_SIGNALS_RETRIEVED = "dtr.social.signals.retrieved.v1"
    PERCEPTION_INTERPRET_RETRIEVED = "dtr.perception.interpret.retrieved.v1"
    CONTEXT_ASSEMBLED = "dtr.context.assembled.v1"
    RESPONSE_CANDIDATES = "dtr.response.candidates.v1"
    RESPONSE_RANKED = "dtr.response.ranked.v1"
    PERCEPTION_EMBEDDINGS = "dtr.perception.embeddings.v1"
    PERCEPTION_EXTRACT = "dtr.perception.extract.v1"
    SOCIAL_PERCEPTION = "dtr.social.perception.v1"
    OUTCOME_SIGNAL = "dtr.feedback.outcome_signal.v1"
    CORRECTION_SIGNAL = "dtr.feedback.correction_signal.v1"
    USER_SUMMARY_REFRESH = "dtr.memory.user_summary.refresh.v1"


LEGACY_SUBJECT_MAP: Dict[str, str] = {
    "dtr.input.received": CanonicalSubjects.INPUT_RECEIVED,
    "dtr.memory.retrieved": CanonicalSubjects.MEMORY_RETRIEVED,
    "dtr.memory.retrieval.requested": CanonicalSubjects.MEMORY_RETRIEVAL_REQUESTED,
    "dtr.social.signals.requested": CanonicalSubjects.SOCIAL_SIGNALS_REQUESTED,
    "dtr.perception.interpret.requested": CanonicalSubjects.PERCEPTION_INTERPRET_REQUESTED,
    "dtr.social.signals.retrieved": CanonicalSubjects.SOCIAL_SIGNALS_RETRIEVED,
    "dtr.perception.interpret.retrieved": CanonicalSubjects.PERCEPTION_INTERPRET_RETRIEVED,
    "dtr.context.assembled": CanonicalSubjects.CONTEXT_ASSEMBLED,
    "dtr.response.candidates": CanonicalSubjects.RESPONSE_CANDIDATES,
    "dtr.response.ranked": CanonicalSubjects.RESPONSE_RANKED,
    "dtr.llm.response_generated": CanonicalSubjects.RESPONSE_RANKED,
    "dtr.perception.embeddings": CanonicalSubjects.PERCEPTION_EMBEDDINGS,
    "dtr.perception.extract": CanonicalSubjects.PERCEPTION_EXTRACT,
    "dtr.social.perception": CanonicalSubjects.SOCIAL_PERCEPTION,
    "dtr.feedback.outcome_signal": CanonicalSubjects.OUTCOME_SIGNAL,
    "dtr.feedback.correction_signal": CanonicalSubjects.CORRECTION_SIGNAL,
    "dtr.memory.user_summary.refresh": CanonicalSubjects.USER_SUMMARY_REFRESH,
}


@dataclass(frozen=True)
class EventEnvelope:
    schema_version: str
    event_id: str
    trace_id: str
    causation_id: str
    created_at: str
    producer: str
    subject: str
    payload: Dict[str, Any]

    @classmethod
    def build(
        cls,
        *,
        subject: str,
        payload: Dict[str, Any],
        producer: str,
        trace_id: str | None = None,
        causation_id: str | None = None,
        schema_version: str = "1.0.0",
    ) -> "EventEnvelope":
        event_id = str(uuid4())
        return cls(
            schema_version=schema_version,
            event_id=event_id,
            trace_id=trace_id or str(uuid4()),
            causation_id=causation_id or event_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            producer=producer,
            subject=to_canonical_subject(subject),
            payload=payload,
        )


def _expect_type(data: dict[str, Any], key: str, expected: type, optional: bool = False) -> Any:
    value = data.get(key)
    if value is None:
        if optional:
            return None
        raise ValueError(f"Missing required key '{key}'")
    if not isinstance(value, expected):
        raise ValueError(f"Field '{key}' must be {expected.__name__}")
    return value


def _expect_uuid(value: str, field_name: str) -> None:
    try:
        UUID(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Field '{field_name}' must be a valid UUID") from exc


def validate_envelope(data: Dict[str, Any]) -> EventEnvelope:
    schema_version = _expect_type(data, "schema_version", str)
    event_id = _expect_type(data, "event_id", str)
    trace_id = _expect_type(data, "trace_id", str)
    causation_id = _expect_type(data, "causation_id", str)
    created_at = _expect_type(data, "created_at", str)
    producer = _expect_type(data, "producer", str)
    subject = _expect_type(data, "subject", str)
    payload = _expect_type(data, "payload", dict)

    _expect_uuid(event_id, "event_id")
    _expect_uuid(trace_id, "trace_id")
    _expect_uuid(causation_id, "causation_id")
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Field 'created_at' must be ISO-8601") from exc

    return EventEnvelope(
        schema_version=schema_version,
        event_id=event_id,
        trace_id=trace_id,
        causation_id=causation_id,
        created_at=created_at,
        producer=producer,
        subject=to_canonical_subject(subject),
        payload=payload,
    )


def to_canonical_subject(subject: str) -> str:
    return LEGACY_SUBJECT_MAP.get(subject, subject)


def validate_cross_service_envelope(subject: str, data: Dict[str, Any]) -> EventEnvelope:
    """Validate mandatory envelope contract for cross-service events."""

    envelope = validate_envelope(data)
    canonical_subject = to_canonical_subject(subject)
    if envelope.subject != canonical_subject:
        raise ValueError(
            f"Envelope subject mismatch: expected '{canonical_subject}', got '{envelope.subject}'"
        )
    return envelope


def normalize_legacy_payload(subject: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    canonical = to_canonical_subject(subject)
    if canonical == CanonicalSubjects.INPUT_RECEIVED and "user_input" not in payload and "text" in payload:
        payload = {**payload, "user_input": payload["text"]}
    if canonical == CanonicalSubjects.MEMORY_RETRIEVED and "retrieved_knowledge" not in payload and "memory" in payload:
        payload = {**payload, "retrieved_knowledge": payload["memory"]}
    if canonical == CanonicalSubjects.RESPONSE_RANKED and "final_response" not in payload and "response" in payload:
        payload = {**payload, "final_response": payload["response"]}
    if canonical == CanonicalSubjects.PERCEPTION_EXTRACT and "text_tokens" not in payload and "tokens" in payload:
        payload = {**payload, "text_tokens": payload["tokens"]}
    return payload


def validate_input_received_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    _expect_type(data, "user_input", str)
    for optional_text_field in ("message_id", "channel_id", "guild_id", "author_id", "reference_message_id", "thread_id"):
        value = data.get(optional_text_field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Field '{optional_text_field}' must be a string")
    attachments = data.get("attachments")
    if attachments is not None:
        if not isinstance(attachments, list):
            raise ValueError("Field 'attachments' must be a list")
        for idx, item in enumerate(attachments):
            if not isinstance(item, dict):
                raise ValueError(f"Attachment at index {idx} must be an object")
            url = item.get("url")
            if not isinstance(url, str) or not url.strip():
                raise ValueError(f"Attachment at index {idx} has invalid url")
            if "size" in item and item["size"] is not None and not isinstance(item["size"], int):
                raise ValueError(f"Attachment at index {idx} has invalid size")
    return data


def validate_memory_retrieved_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    _expect_type(data, "retrieved_knowledge", dict)
    return data


def validate_response_candidates_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    candidates = _expect_type(data, "candidates", list)
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate at index {idx} must be an object")
        _expect_type(candidate, "text", str)
        confidence = candidate.get("confidence", 0.0)
        if not isinstance(confidence, (float, int)):
            raise ValueError(f"Candidate at index {idx} has invalid confidence")

    for optional_object_field in ("interaction_policy", "context_confidence", "social_intent_hints"):
        value = data.get(optional_object_field)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"Field '{optional_object_field}' must be an object")

    affinity = data.get("user_history_affinity")
    if affinity is not None:
        if not isinstance(affinity, dict):
            raise ValueError("Field 'user_history_affinity' must be an object")
        for key, value in affinity.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Field 'user_history_affinity.{key}' must be numeric")
    return data


def validate_response_ranked_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    _expect_type(data, "final_response", str)
    for optional_text_field in ("reply_to_message_id", "thread_id"):
        value = data.get(optional_text_field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Field '{optional_text_field}' must be a string")
    policy = data.get("interaction_policy")
    if policy is not None and not isinstance(policy, dict):
        raise ValueError("Field 'interaction_policy' must be an object")
    if "candidates" in data:
        validate_response_candidates_payload({"candidates": data["candidates"]})
    return data


def validate_perception_embeddings_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    _expect_type(data, "message_id", str)
    _expect_type(data, "user_id", str)
    if "by_modality" in data and not isinstance(data["by_modality"], dict):
        raise ValueError("Field 'by_modality' must be an object")
    return data


def validate_perception_extract_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    _expect_type(data, "message_id", str)
    _expect_type(data, "user_id", str)
    tokens = data.get("text_tokens")
    if tokens is not None and not isinstance(tokens, list):
        raise ValueError("Field 'text_tokens' must be a list")
    return data


PAYLOAD_VALIDATORS = {
    CanonicalSubjects.INPUT_RECEIVED: validate_input_received_payload,
    CanonicalSubjects.MEMORY_RETRIEVED: validate_memory_retrieved_payload,
    CanonicalSubjects.RESPONSE_CANDIDATES: validate_response_candidates_payload,
    CanonicalSubjects.RESPONSE_RANKED: validate_response_ranked_payload,
    CanonicalSubjects.PERCEPTION_EMBEDDINGS: validate_perception_embeddings_payload,
    CanonicalSubjects.PERCEPTION_EXTRACT: validate_perception_extract_payload,
}


def decode_payload(subject: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    canonical = to_canonical_subject(subject)
    normalized = normalize_legacy_payload(canonical, payload)
    validator = PAYLOAD_VALIDATORS.get(canonical)
    if validator is None:
        return normalized
    return validator(normalized)


def decode_payload_or_envelope(subject: str, data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Decode legacy raw payloads and canonical envelopes during migration.

    Returns ``(payload, metadata)`` where metadata may contain envelope fields:
    ``event_id``, ``trace_id``, ``causation_id``, ``producer``, and ``created_at``.
    """

    if "payload" in data and "schema_version" in data:
        envelope = validate_envelope(data)
        payload = decode_payload(subject, envelope.payload)
        return payload, {
            "event_id": envelope.event_id,
            "trace_id": envelope.trace_id,
            "causation_id": envelope.causation_id,
            "producer": envelope.producer,
            "created_at": envelope.created_at,
            "subject": envelope.subject,
        }

    payload = decode_payload(subject, data)
    trace_id = payload.get("trace_id") if isinstance(payload.get("trace_id"), str) else None
    causation_id = payload.get("causation_id") if isinstance(payload.get("causation_id"), str) else None
    return payload, {
        "event_id": None,
        "trace_id": trace_id,
        "causation_id": causation_id,
        "producer": None,
        "created_at": None,
        "subject": to_canonical_subject(subject),
    }
