from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class CanonicalFact:
    id: str
    subject: str
    predicate: str
    object_value: str | None = None
    object_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: str = ""
    updated_at: str = ""
    dedup_key: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_atom(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def canonical_fact_dedup_key(subject: str, predicate: str, object_value: str | None, object_id: str | None = None) -> str:
    payload = "|".join(
        [
            normalized_atom(subject),
            normalized_atom(predicate),
            normalized_atom(object_id),
            normalized_atom(object_value),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def canonical_fact_id(subject: str, predicate: str, object_value: str | None, object_id: str | None = None) -> str:
    return canonical_fact_dedup_key(subject, predicate, object_value, object_id)


def make_canonical_fact(
    *,
    subject: str,
    predicate: str,
    object_value: str | None = None,
    object_id: str | None = None,
    provenance: dict[str, Any] | None = None,
    confidence: float = 1.0,
    created_at: str | None = None,
    updated_at: str | None = None,
    attributes: dict[str, Any] | None = None,
    id: str | None = None,
) -> CanonicalFact:
    now = utc_now_iso()
    created = created_at or now
    dedup_key = canonical_fact_dedup_key(subject, predicate, object_value, object_id)
    return CanonicalFact(
        id=id or canonical_fact_id(subject, predicate, object_value, object_id),
        subject=str(subject),
        predicate=str(predicate),
        object_value=object_value,
        object_id=object_id,
        provenance=dict(provenance or {}),
        confidence=float(confidence),
        created_at=created,
        updated_at=updated_at or created,
        dedup_key=dedup_key,
        attributes=dict(attributes or {}),
    )


def format_fact_snippet(fact: CanonicalFact) -> str:
    obj = fact.object_value or fact.object_id or ""
    if obj:
        return f"{fact.predicate}: {obj}"
    return fact.predicate
