from __future__ import annotations

from typing import Sequence

from .ontology import freshness_score
from .store import GraphEvidence, GraphEvidenceBundle, GraphMemoryStore


def retrieve_user_context(
    store: GraphMemoryStore, user_id: str, *, limit: int = 8
) -> list[GraphEvidence]:
    return _rank_dedup(store.retrieve_user_evidence(user_id, limit=limit), limit)


def retrieve_topic_context(
    store: GraphMemoryStore, topic: str, *, limit: int = 8
) -> list[GraphEvidence]:
    return _rank_dedup(store.retrieve_topic_evidence(topic, limit=limit), limit)


def retrieve_layered_context(
    store: GraphMemoryStore,
    *,
    user_id: str,
    topic: str,
    limit: int = 8,
) -> dict[str, GraphEvidenceBundle]:
    user_items = _rank_dedup(store.retrieve_user_evidence(user_id, limit=limit), limit)
    topic_items = _rank_dedup(store.retrieve_topic_evidence(topic, limit=limit), limit)
    return {
        "user_context": _bundle("user_context", user_items),
        "topic_context": _bundle("topic_context", topic_items),
    }


def _rank_dedup(items: Sequence[GraphEvidence], limit: int) -> list[GraphEvidence]:
    dedup: dict[str, GraphEvidence] = {}
    for item in items:
        key = item.summary.strip().lower()
        prev = dedup.get(key)
        if prev is None or item.score > prev.score:
            dedup[key] = item

    def _sort_key(i: GraphEvidence) -> tuple[bool, float, bool, bool, float, float]:
        attrs = i.attributes or {}
        salience = float(attrs.get("salience", 0.0) or 0.0)
        is_summary = (
            bool(attrs.get("is_summary"))
            or str(attrs.get("fact_type", "")) == "summary"
        )
        is_long_term = str(attrs.get("memory_tier", "")) == "long_term"
        return (
            is_summary,
            salience,
            is_long_term,
            bool(attrs.get("user_scoped")),
            i.score,
            i.confidence,
        )

    ranked = sorted(dedup.values(), key=_sort_key, reverse=True)
    return ranked[:limit]


def _bundle(layer: str, items: list[GraphEvidence]) -> GraphEvidenceBundle:
    enriched: list[GraphEvidence] = []
    for item in items:
        observed_at = (
            item.provenance.observed_at
            or item.attributes.get("updated_at")
            or item.attributes.get("timestamp")
        )
        ontology_type = str(item.attributes.get("ontology_type") or "evidence")
        fresh = freshness_score(observed_at=observed_at, ontology_type=ontology_type)
        enriched.append(
            GraphEvidence(
                evidence_id=item.evidence_id,
                summary=item.summary,
                entity_id=item.entity_id,
                relation_type=item.relation_type,
                score=round(item.score * fresh, 6),
                confidence=item.confidence,
                provenance=item.provenance,
                freshness=fresh,
                attributes=dict(item.attributes),
            )
        )
    provenance = [item.provenance for item in enriched]
    freshness = round(sum(item.freshness for item in enriched) / len(enriched), 6) if enriched else 0.0
    return GraphEvidenceBundle(
        layer=layer,
        evidences=enriched,
        provenance=provenance,
        freshness_score=freshness,
    )
