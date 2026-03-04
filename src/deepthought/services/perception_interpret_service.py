from __future__ import annotations

import json
import logging
from typing import Any

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, PerceptionEmbeddingsEvent
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class PerceptionInterpretService:
    """Convert perception payloads into compact prompt-ready interpretations."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._embeddings_by_input_id: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _attachment_summary(attachments: Any) -> str | None:
        if not isinstance(attachments, list) or not attachments:
            return None
        counts: dict[str, int] = {}
        for raw_attachment in attachments:
            if not isinstance(raw_attachment, dict):
                continue
            content_type = raw_attachment.get("content_type")
            if isinstance(content_type, str) and "/" in content_type:
                media_type = content_type.split("/", maxsplit=1)[0].strip().lower()
            else:
                media_type = "file"
            counts[media_type] = counts.get(media_type, 0) + 1
        if not counts:
            return None
        parts = [f"{media}:{count}" for media, count in sorted(counts.items())]
        return f"attachments[{', '.join(parts)}]"

    @staticmethod
    def _embedding_summary(embeddings_payload: dict[str, Any]) -> dict[str, str]:
        raw_modalities = embeddings_payload.get("by_modality")
        modalities = raw_modalities if isinstance(raw_modalities, dict) else {}
        raw_modality_conf = embeddings_payload.get("modality_confidence")
        modality_conf = raw_modality_conf if isinstance(raw_modality_conf, dict) else {}

        summaries: dict[str, str] = {}
        for modality_name, modality_payload in modalities.items():
            if not isinstance(modality_payload, dict):
                continue
            vectors = modality_payload.get("embeddings")
            span_count = len(modality_payload.get("spans") or [])
            vector_count = len(vectors) if isinstance(vectors, list) else 0
            dim = 0
            if vector_count and isinstance(vectors[0], list):
                dim = len(vectors[0])
            confidence = modality_conf.get(modality_name)
            conf_text = (
                f", conf={float(confidence):.2f}"
                if isinstance(confidence, (int, float))
                else ""
            )
            summaries[str(modality_name)] = (
                f"{modality_name}: {vector_count} vectors"
                f" @dim{dim}, spans={span_count}{conf_text}"
            )
        return summaries

    async def _handle_embeddings(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("Embeddings payload must be an object")
            event = PerceptionEmbeddingsEvent.from_dict(data)
            payload = event.payload
            if payload is None or not payload.input_id:
                await msg.ack()
                return
            self._embeddings_by_input_id[payload.input_id] = {
                "confidence": payload.confidence,
                "modality_confidence": payload.modality_confidence,
                "by_modality": {
                    name: {
                        "spans": mod.spans,
                        "embeddings": mod.embeddings,
                    }
                    for name, mod in payload.by_modality.items()
                },
            }
            await msg.ack()
        except Exception:
            logger.exception("Failed to process PERCEPTION_EMBEDDINGS")
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def _handle_interpret_request(self, msg: Msg) -> None:
        try:
            payload = json.loads(msg.data.decode())
            if not isinstance(payload, dict):
                raise ValueError("Interpret request payload must be an object")
            input_id = payload.get("input_id")
            if not isinstance(input_id, str):
                raise ValueError("Interpret request missing input_id")

            cached = self._embeddings_by_input_id.get(input_id, {})
            modality_summaries = self._embedding_summary(cached)
            attachments_summary = self._attachment_summary(payload.get("attachments"))

            compact_lines: list[str] = []
            if modality_summaries:
                compact_lines.extend(modality_summaries.values())
            if attachments_summary:
                compact_lines.append(attachments_summary)
            if not compact_lines:
                compact_lines.append("no multimodal signals")

            out_payload = {
                "input_id": input_id,
                "multimodal_interpretations": {
                    "summary": " | ".join(compact_lines),
                    "by_modality": modality_summaries,
                    "attachments": attachments_summary,
                },
            }
            await self._publisher.publish(
                EventSubjects.PERCEPTION_INTERPRET_RETRIEVED,
                out_payload,
                use_jetstream=True,
            )
            await msg.ack()
        except Exception:
            logger.exception("Failed to process PERCEPTION_INTERPRET_REQUESTED")
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def start(self, durable_name: str = "perception_interpret_service") -> bool:
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.PERCEPTION_EMBEDDINGS,
                handler=self._handle_embeddings,
                use_jetstream=True,
                durable=f"{durable_name}_embeddings",
            )
            await self._subscriber.subscribe(
                subject=EventSubjects.PERCEPTION_INTERPRET_REQUESTED,
                handler=self._handle_interpret_request,
                use_jetstream=True,
                durable=f"{durable_name}_requests",
            )
            return True
        except nats.errors.Error:
            logger.exception("Failed to start PerceptionInterpretService")
            return False

    async def stop(self) -> None:
        await self._subscriber.unsubscribe_all()

    async def __aenter__(self) -> "PerceptionInterpretService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
