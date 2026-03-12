from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Sequence

from ...graph.connector import GraphConnector, Neo4jConnector
from ...fact_schema import format_fact_snippet
from .store import (
    GraphEntity,
    GraphEvidence,
    GraphFact,
    GraphMemoryStore,
    GraphRelation,
)


class CypherGraphMemoryStore(GraphMemoryStore):
    """Neo4j/Memgraph graph-memory adapter backed by Cypher queries."""

    def __init__(self, connector: GraphConnector | Neo4jConnector) -> None:
        self._connector = connector

    def upsert_entity(self, entity: GraphEntity) -> None:
        self._connector.execute(
            "MERGE (e:Entity {id: $entity_id}) "
            "SET e.type = $entity_type, e.label = $label, e.attributes = $attributes, "
            "e.confidence = $confidence, e.valid_from = $valid_from, e.valid_to = $valid_to, "
            "e.provenance = $provenance",
            {
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type,
                "label": entity.label,
                "attributes": entity.attributes,
                "confidence": entity.confidence,
                "valid_from": entity.temporal.valid_from,
                "valid_to": entity.temporal.valid_to,
                "provenance": asdict(entity.provenance),
            },
        )

    def upsert_relation(self, relation: GraphRelation) -> None:
        self._connector.execute(
            "MERGE (s:Entity {id: $source_id}) "
            "MERGE (t:Entity {id: $target_id}) "
            "MERGE (s)-[r:RELATION {type: $relation_type}]->(t) "
            "SET r.attributes = $attributes, r.confidence = $confidence, "
            "r.valid_from = $valid_from, r.valid_to = $valid_to, r.provenance = $provenance",
            {
                "source_id": relation.source_id,
                "target_id": relation.target_id,
                "relation_type": relation.relation_type,
                "attributes": relation.attributes,
                "confidence": relation.confidence,
                "valid_from": relation.temporal.valid_from,
                "valid_to": relation.temporal.valid_to,
                "provenance": asdict(relation.provenance),
            },
        )

    def upsert_fact(self, fact: GraphFact) -> None:
        self._connector.execute(
            "MERGE (s:Entity {id: $subject}) "
            "MERGE (f:Fact {dedup_key: $dedup_key}) "
            "ON CREATE SET f.id = $id, f.created_at = $created_at "
            "SET f.subject = $subject, f.predicate = $predicate, f.attributes = $attributes, "
            "f.confidence = $confidence, f.updated_at = $updated_at, f.provenance = $provenance, "
            "f.object_value = $object_value, f.object_id = $object_id, f.dedup_key = $dedup_key "
            "MERGE (s)-[:HAS_FACT]->(f)",
            {
                "id": fact.id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object_id": fact.object_id,
                "object_value": fact.object_value,
                "attributes": fact.attributes,
                "confidence": fact.confidence,
                "created_at": fact.created_at,
                "updated_at": fact.updated_at,
                "provenance": fact.provenance,
                "dedup_key": fact.dedup_key,
            },
        )
        if fact.object_id:
            self._connector.execute(
                "MERGE (o:Entity {id: $object_id}) "
                "MERGE (f:Fact {dedup_key: $dedup_key}) "
                "MERGE (f)-[:ABOUT]->(o)",
                {"object_id": fact.object_id, "dedup_key": fact.dedup_key},
            )

    def retrieve_user_evidence(
        self, user_id: str, *, limit: int = 10
    ) -> Sequence[GraphEvidence]:
        rows = self._connector.execute(
            "MATCH (u:Entity {id: $user_id})-[:HAS_FACT]->(f:Fact) "
            "OPTIONAL MATCH (f)-[:ABOUT]->(o:Entity) "
            "RETURN coalesce(f.id, f.dedup_key) AS evidence_id, f.subject AS subject, "
            "f.predicate AS predicate, f.object_value AS object_value, "
            "o.id AS object_id, f.confidence AS confidence, f.provenance AS provenance, "
            "f.attributes AS attributes, f.created_at AS created_at, f.updated_at AS updated_at "
            "ORDER BY f.confidence DESC, f.updated_at DESC LIMIT $limit",
            {"user_id": user_id, "limit": limit},
        )
        return _rows_to_evidence(rows)

    def retrieve_topic_evidence(
        self, topic: str, *, limit: int = 10
    ) -> Sequence[GraphEvidence]:
        rows = self._connector.execute(
            "MATCH (s:Entity)-[:HAS_FACT]->(f:Fact) "
            "WHERE toLower(f.predicate) CONTAINS toLower($topic) "
            "OR toLower(coalesce(f.object_value, '')) CONTAINS toLower($topic) "
            "RETURN coalesce(f.id, f.dedup_key) AS evidence_id, f.subject AS subject, "
            "f.predicate AS predicate, f.object_value AS object_value, "
            "f.object_id AS object_id, f.confidence AS confidence, f.provenance AS provenance, "
            "f.attributes AS attributes, f.created_at AS created_at, f.updated_at AS updated_at "
            "ORDER BY f.confidence DESC, f.updated_at DESC LIMIT $limit",
            {"topic": topic, "limit": limit},
        )
        return _rows_to_evidence(rows)


class InMemoryGraphMemoryStore(GraphMemoryStore):
    """Local fallback adapter used in tests and development."""

    def __init__(self) -> None:
        self.entities: dict[str, GraphEntity] = {}
        self.relations: dict[tuple[str, str, str], GraphRelation] = {}
        self.facts: dict[str, GraphFact] = {}

    def upsert_entity(self, entity: GraphEntity) -> None:
        self.entities[entity.entity_id] = entity

    def upsert_relation(self, relation: GraphRelation) -> None:
        self.relations[
            (relation.source_id, relation.relation_type, relation.target_id)
        ] = relation

    def upsert_fact(self, fact: GraphFact) -> None:
        prev = self.facts.get(fact.dedup_key)
        if prev is None or fact.updated_at >= prev.updated_at:
            self.facts[fact.dedup_key] = fact

    def retrieve_user_evidence(
        self, user_id: str, *, limit: int = 10
    ) -> Sequence[GraphEvidence]:
        out = [
            _fact_to_evidence(f)
            for f in self.facts.values()
            if f.subject == user_id or f.object_id == user_id
        ]
        return _rank_and_dedup(out, limit)

    def retrieve_topic_evidence(
        self, topic: str, *, limit: int = 10
    ) -> Sequence[GraphEvidence]:
        topic_l = topic.lower()
        out = []
        for fact in self.facts.values():
            hay = " ".join(
                [
                    fact.predicate.lower(),
                    str(fact.object_value or "").lower(),
                    str(fact.attributes.get("topic", "")).lower(),
                ]
            )
            if topic_l in hay:
                out.append(_fact_to_evidence(fact))
        return _rank_and_dedup(out, limit)


def _rows_to_evidence(rows: Sequence[Any]) -> list[GraphEvidence]:
    evidences = []
    for row in rows:
        evidence_id = _row_get(row, "evidence_id", "")
        subject = _row_get(row, "subject", "")
        predicate = _row_get(row, "predicate", "fact")
        object_value = _row_get(row, "object_value", "")
        confidence = float(_row_get(row, "confidence", 0.5) or 0.5)
        provenance = _row_get(row, "provenance", {}) or {}
        attrs = _row_get(row, "attributes", {}) or {}
        attrs = dict(attrs)
        attrs.setdefault("subject", subject)
        attrs.setdefault("user_scoped", bool(subject))
        attrs.setdefault("created_at", _row_get(row, "created_at"))
        attrs.setdefault("updated_at", _row_get(row, "updated_at"))
        summary = f"{predicate}: {object_value}".strip()
        evidences.append(
            GraphEvidence(
                evidence_id=str(evidence_id),
                summary=summary,
                entity_id=_row_get(row, "object_id"),
                relation_type=predicate,
                score=_score(confidence, attrs),
                confidence=confidence,
                provenance=_prov_from_any(provenance),
                attributes=attrs,
            )
        )
    return _rank_and_dedup(evidences, limit=len(evidences))


def _prov_from_any(raw: Any):
    from .store import Provenance

    if isinstance(raw, dict):
        return Provenance(
            source=str(raw.get("source", "unknown")),
            source_id=raw.get("source_id"),
            observed_at=raw.get("observed_at"),
        )
    return Provenance(source="unknown")


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _fact_to_evidence(fact: GraphFact) -> GraphEvidence:
    attrs = dict(fact.attributes)
    attrs.setdefault("subject", fact.subject)
    attrs.setdefault("user_scoped", True)
    attrs.setdefault("created_at", fact.created_at)
    attrs.setdefault("updated_at", fact.updated_at)
    return GraphEvidence(
        evidence_id=fact.id,
        summary=format_fact_snippet(fact),
        entity_id=fact.object_id,
        relation_type=fact.predicate,
        score=_score(fact.confidence, fact.attributes),
        confidence=fact.confidence,
        provenance=_prov_from_any(fact.provenance),
        attributes=attrs,
    )


def _score(confidence: float, attrs: dict[str, Any]) -> float:
    recency_boost = 0.0
    observed = attrs.get("observed_at") or attrs.get("updated_at") or attrs.get("timestamp")
    if observed:
        try:
            delta = datetime.now(timezone.utc) - datetime.fromisoformat(
                str(observed).replace("Z", "+00:00")
            )
            recency_boost = max(0.0, 1.0 - min(delta.days / 365.0, 1.0)) * 0.1
        except Exception:
            recency_boost = 0.0
    salience = float(attrs.get("salience", 0.0) or 0.0) * 0.3
    return round(float(confidence) + recency_boost + salience, 6)


def _rank_and_dedup(
    evidences: Sequence[GraphEvidence], limit: int
) -> list[GraphEvidence]:
    dedup: dict[str, GraphEvidence] = {}
    for item in evidences:
        key = item.summary.strip().lower()
        prev = dedup.get(key)
        if prev is None or item.score > prev.score:
            dedup[key] = item
    ranked = sorted(dedup.values(), key=lambda x: (x.score, x.confidence), reverse=True)
    return ranked[:limit]
