from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from typing import Any

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, PerceptionEmbeddingsEvent
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from .perception.summarization import build_semantic_notes

logger = logging.getLogger(__name__)


class PerceptionInterpretService:
    """Convert perception payloads into compact prompt-ready interpretations."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        *,
        cache_max_entries: int = 512,
        cache_max_age_seconds: float = 300.0,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._cache_max_entries = max(1, cache_max_entries)
        self._cache_max_age_seconds = max(0.0, cache_max_age_seconds)
        self._embeddings_by_input_id: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    def _evict(self, input_id: str) -> None:
        if self._embeddings_by_input_id.pop(input_id, None) is not None:
            self._cache_evictions += 1

    def _prune_expired(self, now: float | None = None) -> None:
        if not self._embeddings_by_input_id:
            return
        ts = time.monotonic() if now is None else now
        expired: list[str] = []
        for input_id, entry in self._embeddings_by_input_id.items():
            if (ts - float(entry.get("cached_at", ts))) > self._cache_max_age_seconds:
                expired.append(input_id)
        for input_id in expired:
            self._evict(input_id)

    def _cache_get(self, input_id: str) -> dict[str, Any]:
        self._prune_expired()
        entry = self._embeddings_by_input_id.get(input_id)
        if entry is None:
            self._cache_misses += 1
            return {}
        self._cache_hits += 1
        self._embeddings_by_input_id.move_to_end(input_id)
        return entry.get("payload", {})

    def _cache_put(self, input_id: str, payload: dict[str, Any]) -> None:
        self._prune_expired()
        self._embeddings_by_input_id[input_id] = {
            "payload": payload,
            "cached_at": time.monotonic(),
        }
        self._embeddings_by_input_id.move_to_end(input_id)
        while len(self._embeddings_by_input_id) > self._cache_max_entries:
            oldest_input_id = next(iter(self._embeddings_by_input_id))
            self._evict(oldest_input_id)

    @property
    def cache_metrics(self) -> dict[str, float | int]:
        total_lookups = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_lookups) if total_lookups else 0.0
        return {
            "cache_size": len(self._embeddings_by_input_id),
            "evictions": self._cache_evictions,
            "hit_rate": hit_rate,
        }

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
            self._cache_put(
                payload.input_id,
                {
                "confidence": payload.confidence,
                "modality_confidence": payload.modality_confidence,
                "by_modality": {
                    name: {
                        "spans": mod.spans,
                        "embeddings": mod.embeddings,
                    }
                    for name, mod in payload.by_modality.items()
                },
                },
            )
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

            cached = self._cache_get(input_id)
            multimodal_notes = build_semantic_notes(
                attachments=payload.get("attachments"),
                embeddings_payload=cached,
            )

            out_payload = {
                "input_id": input_id,
                "multimodal_interpretations": multimodal_notes,
            }
            await self._publisher.publish(
                EventSubjects.PERCEPTION_INTERPRET_RETRIEVED,
                out_payload,
                use_jetstream=True,
            )
            self._evict(input_id)
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
