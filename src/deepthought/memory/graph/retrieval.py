from __future__ import annotations

from typing import Sequence

from .store import GraphEvidence, GraphMemoryStore


def retrieve_user_context(store: GraphMemoryStore, user_id: str, *, limit: int = 8) -> list[GraphEvidence]:
    return _rank_dedup(store.retrieve_user_evidence(user_id, limit=limit), limit)


def retrieve_topic_context(store: GraphMemoryStore, topic: str, *, limit: int = 8) -> list[GraphEvidence]:
    return _rank_dedup(store.retrieve_topic_evidence(topic, limit=limit), limit)


def _rank_dedup(items: Sequence[GraphEvidence], limit: int) -> list[GraphEvidence]:
    dedup: dict[str, GraphEvidence] = {}
    for item in items:
        key = item.summary.strip().lower()
        prev = dedup.get(key)
        if prev is None or item.score > prev.score:
            dedup[key] = item
    ranked = sorted(dedup.values(), key=lambda i: (i.score, i.confidence), reverse=True)
    return ranked[:limit]
