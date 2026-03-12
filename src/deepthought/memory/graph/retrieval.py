from __future__ import annotations

from typing import Sequence

from .store import GraphEvidence, GraphMemoryStore


def retrieve_user_context(
    store: GraphMemoryStore, user_id: str, *, limit: int = 8
) -> list[GraphEvidence]:
    return _rank_dedup(store.retrieve_user_evidence(user_id, limit=limit), limit)


def retrieve_topic_context(
    store: GraphMemoryStore, topic: str, *, limit: int = 8
) -> list[GraphEvidence]:
    return _rank_dedup(store.retrieve_topic_evidence(topic, limit=limit), limit)


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
