from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from .store import TemporalValidity

ONTOLOGY_TYPES = {
    "user",
    "preference",
    "commitment",
    "event",
    "relationship",
    "topic",
    "evidence",
}


@dataclass(frozen=True)
class TypedPredicate:
    name: str
    subject_type: str
    object_type: str
    ontology_type: str


PREDICATE_REGISTRY: dict[str, TypedPredicate] = {
    "has_nickname": TypedPredicate("has_nickname", "user", "topic", "relationship"),
    "likes_hobby": TypedPredicate("likes_hobby", "user", "preference", "preference"),
    "favorite": TypedPredicate("favorite", "user", "preference", "preference"),
    "plans_event": TypedPredicate("plans_event", "user", "event", "commitment"),
    "mentioned": TypedPredicate("mentioned", "user", "evidence", "evidence"),
    "memory_note": TypedPredicate("memory_note", "user", "evidence", "evidence"),
    "multimodal_summary": TypedPredicate("multimodal_summary", "user", "evidence", "evidence"),
}

HALF_LIFE_DAYS = {
    "preference": 365.0,
    "commitment": 14.0,
    "event": 21.0,
    "relationship": 90.0,
    "topic": 120.0,
    "evidence": 45.0,
    "user": 720.0,
}


def predicate_spec(predicate: str, fallback_type: str = "evidence") -> TypedPredicate:
    spec = PREDICATE_REGISTRY.get(predicate)
    if spec is not None:
        return spec
    return TypedPredicate(predicate, "user", fallback_type, fallback_type)


def classify_ontology_type(*, fact_type: str, predicate: str, object_type: str | None = None) -> str:
    spec = PREDICATE_REGISTRY.get(predicate)
    if spec is not None:
        return spec.ontology_type
    if fact_type in {"preference", "profile", "temporal_fact", "utterance"}:
        return {
            "preference": "preference",
            "profile": "relationship",
            "temporal_fact": "commitment",
            "utterance": "evidence",
        }[fact_type]
    if object_type in ONTOLOGY_TYPES:
        return str(object_type)
    return "evidence"


def confidence_with_decay(
    confidence: float,
    *,
    observed_at: str | None,
    ontology_type: str,
    now: datetime | None = None,
) -> float:
    freshness = freshness_score(observed_at=observed_at, ontology_type=ontology_type, now=now)
    return round(float(confidence) * freshness, 6)


def freshness_score(*, observed_at: str | None, ontology_type: str, now: datetime | None = None) -> float:
    if not observed_at:
        return 0.75
    try:
        point = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
        ref = now or datetime.now(timezone.utc)
        age_days = max(0.0, (ref - point).total_seconds() / 86400.0)
    except ValueError:
        return 0.75

    half_life = HALF_LIFE_DAYS.get(ontology_type, 60.0)
    if half_life <= 0:
        return 1.0
    decay = 0.5 ** (age_days / half_life)
    return round(max(0.05, min(1.0, decay)), 6)


def temporal_validity(timestamp: str | None, ontology_type: str) -> TemporalValidity:
    if ontology_type == "commitment" and timestamp:
        return TemporalValidity(valid_from=timestamp)
    return TemporalValidity(valid_from=timestamp)


def relation_projection_name(ontology_type: str, predicate: str) -> str:
    if ontology_type in {"preference", "commitment", "relationship", "topic"}:
        return ontology_type
    return predicate
