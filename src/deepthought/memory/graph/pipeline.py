from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

from ..fact_extractor import extract_typed_fact_triples_from_turn
from ...fact_schema import make_canonical_fact
from .ontology import (
    classify_ontology_type,
    confidence_with_decay,
    predicate_spec,
    relation_projection_name,
    temporal_validity,
)
from .store import (
    ConversationalGraphObject,
    GraphEntity,
    GraphFact,
    GraphMemoryStore,
    GraphRelation,
    Provenance,
    utc_now_iso,
)


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
    """Write pre-extracted triples by projecting typed conversational objects."""

    objects = triples_to_graph_objects(triples, timestamp=timestamp, source_id=source_id)
    return ingest_graph_objects(objects, store)


def triples_to_graph_objects(
    triples: Sequence[dict[str, Any]],
    *,
    timestamp: str,
    source_id: str,
) -> list[ConversationalGraphObject]:
    objects: list[ConversationalGraphObject] = []
    for triple in triples:
        ontology_type = classify_ontology_type(
            fact_type=str(triple.get("fact_type", "")),
            predicate=str(triple.get("predicate", "")),
            object_type=str(triple.get("object_type", "")),
        )
        obj_id = triple.get("object_id")
        if not obj_id and str(triple.get("fact_type")) == "temporal_fact" and triple.get("object_value"):
            obj_id = f"event:{_stable_id(triple['subject_id'], str(triple.get('object_value')))}"
        confidence = float(triple.get("confidence", 0.7))
        attributes = {
            **dict(triple.get("attributes", {})),
            "fact_type": triple.get("fact_type", "profile"),
            "subject_type": triple.get("subject_type", "user"),
            "object_type": triple.get("object_type", ontology_type),
            "timestamp": timestamp,
        }
        provenance = Provenance(source="conversation_turn", source_id=source_id, observed_at=timestamp)
        objects.append(
            ConversationalGraphObject(
                object_id=f"obj:{_stable_id(triple['subject_id'], str(triple.get('predicate')), str(obj_id), str(triple.get('object_value')))}",
                ontology_type=ontology_type,
                subject_id=triple["subject_id"],
                predicate=triple["predicate"],
                object_id_ref=obj_id,
                object_value=str(triple.get("object_value")) if triple.get("object_value") is not None else None,
                confidence=confidence_with_decay(
                    confidence, observed_at=timestamp, ontology_type=ontology_type
                ),
                provenance=provenance,
                attributes=attributes,
                temporal=temporal_validity(timestamp, ontology_type),
            )
        )
    return objects


def ingest_graph_objects(
    objects: Sequence[ConversationalGraphObject],
    store: GraphMemoryStore,
) -> int:
    """Project typed graph objects into entities/relations/facts."""

    upserts = 0
    for obj in objects:
        spec = predicate_spec(obj.predicate, fallback_type=obj.ontology_type)
        subject = GraphEntity(
            entity_id=obj.subject_id,
            entity_type=spec.subject_type,
            label=obj.subject_id,
            attributes={"last_seen": obj.provenance.observed_at, "ontology_type": "user"},
            provenance=obj.provenance,
            confidence=obj.confidence,
        )
        store.upsert_entity(subject)

        if obj.object_id_ref:
            store.upsert_entity(
                GraphEntity(
                    entity_id=obj.object_id_ref,
                    entity_type=spec.object_type,
                    label=obj.object_id_ref,
                    attributes={
                        **obj.attributes,
                        "ontology_type": obj.ontology_type,
                    },
                    provenance=subject.provenance,
                    confidence=obj.confidence,
                )
            )
            store.upsert_relation(
                GraphRelation(
                    source_id=obj.subject_id,
                    relation_type=relation_projection_name(obj.ontology_type, obj.predicate),
                    target_id=obj.object_id_ref,
                    attributes={**obj.attributes, "predicate": obj.predicate, "ontology_type": obj.ontology_type},
                    provenance=subject.provenance,
                    confidence=obj.confidence,
                    temporal=obj.temporal,
                )
            )

        canonical = make_canonical_fact(
            subject=obj.subject_id,
            predicate=obj.predicate,
            object_id=obj.object_id_ref,
            object_value=obj.object_value,
            provenance={
                "source": obj.provenance.source,
                "source_id": obj.provenance.source_id,
                "observed_at": obj.provenance.observed_at,
            },
            confidence=obj.confidence,
            created_at=obj.provenance.observed_at,
            updated_at=obj.provenance.observed_at,
            attributes={
                **obj.attributes,
                "ontology_type": obj.ontology_type,
            },
        )
        store.upsert_fact(GraphFact(**canonical.__dict__))
        upserts += 1
    return upserts


def _stable_id(*parts: Iterable[str] | str) -> str:
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()
