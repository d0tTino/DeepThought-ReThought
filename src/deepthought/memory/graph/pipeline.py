from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

from ..fact_extractor import extract_typed_fact_triples_from_turn
from .store import GraphEntity, GraphFact, GraphMemoryStore, GraphRelation, Provenance, TemporalValidity, utc_now_iso


def ingest_conversation_turns(
    turns: Sequence[dict[str, Any]],
    store: GraphMemoryStore,
    *,
    default_user_id: str = "anonymous",
) -> int:
    """Extract typed triples from turns and write entities/relations/facts."""

    upserts = 0
    for turn in turns:
        user_id = str(turn.get("user_id") or default_user_id)
        text = str(turn.get("text") or "")
        timestamp = str(turn.get("timestamp") or utc_now_iso())
        source_id = str(turn.get("input_id") or turn.get("id") or _stable_id(user_id, text, timestamp))

        triples = extract_typed_fact_triples_from_turn(user_id=user_id, message=text, timestamp=timestamp, source_id=source_id)
        upserts += ingest_fact_triples(triples, store, timestamp=timestamp, source_id=source_id)
    return upserts


def ingest_fact_triples(
    triples: Sequence[dict[str, Any]],
    store: GraphMemoryStore,
    *,
    timestamp: str,
    source_id: str,
) -> int:
    """Write pre-extracted triples into graph entities/relations/facts."""

    upserts = 0
    for triple in triples:
        subject = GraphEntity(
            entity_id=triple["subject_id"],
            entity_type=triple.get("subject_type", "user"),
            label=triple.get("subject_label", triple["subject_id"]),
            attributes={"last_seen": timestamp},
            provenance=Provenance(source="conversation_turn", source_id=source_id, observed_at=timestamp),
            confidence=float(triple.get("confidence", 0.7)),
        )
        store.upsert_entity(subject)

        obj_id = triple.get("object_id")
        if not obj_id and triple.get("fact_type") == "temporal_fact" and triple.get("object_value"):
            obj_id = f"event:{_stable_id(triple['subject_id'], str(triple.get('object_value')))}"

        if obj_id:
            store.upsert_entity(
                GraphEntity(
                    entity_id=obj_id,
                    entity_type=triple.get("object_type", "topic"),
                    label=triple.get("object_label", obj_id),
                    attributes=triple.get("object_attributes", {}),
                    provenance=subject.provenance,
                    confidence=float(triple.get("confidence", 0.7)),
                )
            )
            store.upsert_relation(
                GraphRelation(
                    source_id=triple["subject_id"],
                    relation_type=_typed_relation(triple.get("fact_type", "profile"), triple["predicate"]),
                    target_id=obj_id,
                    attributes={**triple.get("attributes", {}), "predicate": triple["predicate"]},
                    provenance=subject.provenance,
                    confidence=float(triple.get("confidence", 0.7)),
                    temporal=TemporalValidity(valid_from=timestamp),
                )
            )
        fact_id = _stable_id(triple["subject_id"], triple["predicate"], str(obj_id or triple.get("object_value")))
        store.upsert_fact(
            GraphFact(
                fact_id=fact_id,
                subject_id=triple["subject_id"],
                predicate=triple["predicate"],
                object_id=obj_id,
                object_value=triple.get("object_value"),
                fact_type=triple.get("fact_type", "profile"),
                attributes={**triple.get("attributes", {}), "timestamp": timestamp},
                provenance=subject.provenance,
                confidence=float(triple.get("confidence", 0.7)),
                temporal=TemporalValidity(valid_from=timestamp),
            )
        )
        upserts += 1
    return upserts


def _typed_relation(fact_type: str, predicate: str) -> str:
    return {
        "preference": "preference",
        "profile": "profile",
        "temporal_fact": "temporal_fact",
    }.get(fact_type, predicate)


def _stable_id(*parts: Iterable[str] | str) -> str:
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()
