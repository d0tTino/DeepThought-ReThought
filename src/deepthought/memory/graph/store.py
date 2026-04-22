from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence

from ...fact_schema import CanonicalFact


@dataclass(frozen=True)
class Provenance:
    source: str
    source_id: str | None = None
    observed_at: str | None = None


@dataclass(frozen=True)
class TemporalValidity:
    valid_from: str | None = None
    valid_to: str | None = None


@dataclass(frozen=True)
class GraphEntity:
    entity_id: str
    entity_type: str
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=lambda: Provenance(source="unknown"))
    confidence: float = 1.0
    temporal: TemporalValidity = field(default_factory=TemporalValidity)


@dataclass(frozen=True)
class GraphRelation:
    source_id: str
    relation_type: str
    target_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=lambda: Provenance(source="unknown"))
    confidence: float = 1.0
    temporal: TemporalValidity = field(default_factory=TemporalValidity)


@dataclass(frozen=True)
class GraphFact(CanonicalFact):
    """Backward-compatible alias for canonical fact records."""


@dataclass(frozen=True)
class GraphEvidence:
    evidence_id: str
    summary: str
    entity_id: str | None
    relation_type: str | None
    score: float
    confidence: float
    provenance: Provenance
    freshness: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEvidenceBundle:
    layer: str
    evidences: list[GraphEvidence]
    provenance: list[Provenance]
    freshness_score: float


@dataclass(frozen=True)
class ConversationalGraphObject:
    object_id: str
    ontology_type: str
    subject_id: str
    predicate: str
    object_id_ref: str | None
    object_value: str | None
    confidence: float
    provenance: Provenance
    attributes: dict[str, Any] = field(default_factory=dict)
    temporal: TemporalValidity = field(default_factory=TemporalValidity)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GraphMemoryStore(Protocol):
    def upsert_entity(self, entity: GraphEntity) -> None:
        ...

    def upsert_relation(self, relation: GraphRelation) -> None:
        ...

    def upsert_fact(self, fact: GraphFact) -> None:
        ...

    def retrieve_user_evidence(self, user_id: str, *, limit: int = 10) -> Sequence[GraphEvidence]:
        ...

    def retrieve_topic_evidence(self, topic: str, *, limit: int = 10) -> Sequence[GraphEvidence]:
        ...
