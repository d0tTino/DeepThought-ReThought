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
from .perception.summarization import build_semantic_notes

logger = logging.getLogger(__name__)


class PerceptionInterpretService:
    """Convert perception payloads into compact prompt-ready interpretations."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._embeddings_by_input_id: dict[str, dict[str, Any]] = {}

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
