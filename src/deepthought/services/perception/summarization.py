from __future__ import annotations

from typing import Any


def _artifact_id_for_attachment(attachments: Any, modality: str, index: int) -> str:
    if isinstance(attachments, list):
        matching_index = 0
        for raw_attachment in attachments:
            if not isinstance(raw_attachment, dict):
                continue
            content_type = raw_attachment.get("content_type")
            media_type = (
                content_type.split("/", maxsplit=1)[0].strip().lower()
                if isinstance(content_type, str) and "/" in content_type
                else "file"
            )
            if media_type != modality:
                continue
            if matching_index == index:
                for key in ("artifact_id", "id", "url", "filename"):
                    value = raw_attachment.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            matching_index += 1
    return f"{modality}:artifact:{index}"


def _span_time_range(modality: str, span: Any) -> tuple[list[int] | None, dict[str, int] | None]:
    if not isinstance(span, (list, tuple)) or len(span) < 2:
        return None, None
    parsed = [int(span[0]), int(span[1])]
    if modality.lower() in {"audio", "video"}:
        return parsed, {"start_ms": parsed[0], "end_ms": parsed[1]}
    return parsed, None


LOW_CONFIDENCE_THRESHOLD = 0.45


def build_semantic_notes(*, attachments: Any, embeddings_payload: dict[str, Any]) -> dict[str, Any]:
    """Build stable multimodal interpretation notes for prompt injection."""

    raw_modalities = embeddings_payload.get("by_modality")
    modalities = raw_modalities if isinstance(raw_modalities, dict) else {}
    raw_modality_conf = embeddings_payload.get("modality_confidence")
    modality_conf = raw_modality_conf if isinstance(raw_modality_conf, dict) else {}

    notes: list[dict[str, Any]] = []
    by_modality: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    confidences: list[float] = []

    for modality_name, modality_payload in modalities.items():
        if not isinstance(modality_payload, dict):
            continue
        vectors = modality_payload.get("embeddings")
        vector_count = len(vectors) if isinstance(vectors, list) else 0
        spans = modality_payload.get("spans")
        span_list = spans if isinstance(spans, list) else []
        span_count = len(span_list)
        encoders = modality_payload.get("encoders")
        encoder_names = [
            str(enc.get("name"))
            for enc in encoders
            if isinstance(enc, dict) and isinstance(enc.get("name"), str)
        ] if isinstance(encoders, list) else []
        confidence = modality_conf.get(modality_name)
        conf = float(confidence) if isinstance(confidence, (int, float)) else 0.0
        confidences.append(conf)

        where_hint = "temporal spans" if str(modality_name).lower() == "audio" else "attachment regions"
        what = f"{vector_count} embedding vectors across {span_count} spans"
        who = "unknown"

        modality = str(modality_name)
        evidence_ids: list[str] = []
        for span_index, span in enumerate(span_list):
            parsed_span, time_range = _span_time_range(modality, span)
            evidence_id = f"{modality}:{span_index}"
            evidence_ids.append(evidence_id)
            uncertainty_reason = "low modality confidence" if conf < LOW_CONFIDENCE_THRESHOLD else ""
            evidence.append({
                "evidence_id": evidence_id,
                "artifact_id": _artifact_id_for_attachment(attachments, modality, span_index),
                "modality": modality,
                "span": parsed_span,
                "time_range": time_range,
                "confidence": round(conf, 3),
                "uncertainty_reason": uncertainty_reason,
                "extraction_method": "embedding_span_summary",
                "encoders": encoder_names,
            })

        note = {
            "modality": modality,
            "what": what,
            "where": where_hint,
            "who": who,
            "confidence": round(conf, 3),
            "evidence_ids": evidence_ids,
        }
        notes.append(note)
        by_modality[str(modality_name)] = note

    attachment_counts: dict[str, int] = {}
    if isinstance(attachments, list):
        for raw_attachment in attachments:
            if not isinstance(raw_attachment, dict):
                continue
            content_type = raw_attachment.get("content_type")
            if isinstance(content_type, str) and "/" in content_type:
                media_type = content_type.split("/", maxsplit=1)[0].strip().lower()
            else:
                media_type = "file"
            attachment_counts[media_type] = attachment_counts.get(media_type, 0) + 1

    attachment_summary = None
    if attachment_counts:
        rendered = ", ".join(f"{kind}:{count}" for kind, count in sorted(attachment_counts.items()))
        attachment_summary = f"attachments[{rendered}]"

    aggregate_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    low_confidence = aggregate_confidence < LOW_CONFIDENCE_THRESHOLD
    summary_parts = [
        f"{note['modality']}: {note['what']} (conf={note['confidence']:.2f})"
        for note in notes
    ]
    if attachment_summary:
        summary_parts.append(attachment_summary)
    summary = " | ".join(summary_parts) if summary_parts else "no multimodal signals"

    return {
        "schema_version": "multimodal.semantic-notes.v1",
        "summary": summary,
        "notes": notes,
        "by_modality": by_modality,
        "attachments": attachment_summary,
        "evidence": evidence,
        "confidence": {
            "aggregate": aggregate_confidence,
            "low_confidence": low_confidence,
            "threshold": LOW_CONFIDENCE_THRESHOLD,
        },
        "fallback": {
            "ask_clarifying_question": low_confidence,
            "reason": "low multimodal confidence" if low_confidence else "",
        },
    }

