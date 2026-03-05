from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import ContextAssembledPayload, EventSubjects
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


@dataclass
class _PendingAssembly:
    request: dict[str, Any]
    provider_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    published: bool = False
    event: asyncio.Event = field(default_factory=asyncio.Event)


class ContextAssemblerService:
    """Assemble contextual data from memory/social/perception providers."""

    _PROVIDER_ORDER = ("memory", "social", "perception")

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        *,
        wait_window_seconds: float = 0.2,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._wait_window_seconds = max(0.01, wait_window_seconds)
        self._pending: dict[str, _PendingAssembly] = {}
        self._lock = asyncio.Lock()

    async def _publish_fanout_requests(self, payload: dict[str, Any]) -> None:
        fanout_payload = {
            "input_id": payload.get("input_id"),
            "user_input": payload.get("user_input"),
            "user_id": payload.get("user_id"),
            "author_id": payload.get("author_id"),
            "channel_id": payload.get("channel_id"),
            "author_name": payload.get("author_name"),
            "timestamp": payload.get("timestamp"),
            "attachments": payload.get("attachments"),
        }
        await self._publisher.publish(EventSubjects.MEMORY_RETRIEVAL_REQUESTED, fanout_payload, use_jetstream=True)
        await self._publisher.publish(EventSubjects.SOCIAL_SIGNALS_REQUESTED, fanout_payload, use_jetstream=True)
        await self._publisher.publish(EventSubjects.PERCEPTION_INTERPRET_REQUESTED, fanout_payload, use_jetstream=True)

    async def _handle_input_received(self, msg: Msg) -> None:
        try:
            payload = json.loads(msg.data.decode())
            if not isinstance(payload, dict):
                raise ValueError("Input payload must be an object")
            input_id = payload.get("input_id")
            user_input = payload.get("user_input")
            if not isinstance(input_id, str) or not isinstance(user_input, str):
                raise ValueError("Input payload missing required fields")

            pending = _PendingAssembly(request=payload)
            async with self._lock:
                self._pending[input_id] = pending

            await self._publish_fanout_requests(payload)
            asyncio.create_task(self._await_and_publish(input_id))
            await msg.ack()
        except Exception:
            logger.exception("Failed to process input event in ContextAssemblerService")
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def _handle_provider_response(self, msg: Msg, provider_name: str) -> None:
        try:
            payload = json.loads(msg.data.decode())
            if not isinstance(payload, dict):
                raise ValueError("Provider payload must be an object")
            input_id = payload.get("input_id")
            if not isinstance(input_id, str):
                raise ValueError("Provider payload missing input_id")

            async with self._lock:
                pending = self._pending.get(input_id)
                if pending is not None and not pending.published:
                    pending.provider_payloads[provider_name] = payload
                    pending.event.set()
            await msg.ack()
        except Exception:
            logger.exception("Failed to process provider response for %s", provider_name)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    def _build_context_payload(self, input_id: str, pending: _PendingAssembly, elapsed: float) -> ContextAssembledPayload:
        request = pending.request
        memory_payload = pending.provider_payloads.get("memory", {})
        retrieved = memory_payload.get("retrieved_knowledge", {})
        facts = retrieved.get("facts") if isinstance(retrieved, dict) else []
        if not isinstance(facts, list):
            facts = []

        social_payload = pending.provider_payloads.get("social", {})
        social_signals = social_payload.get("social_signals", social_payload)
        if not isinstance(social_signals, dict):
            social_signals = {}

        perception_payload = pending.provider_payloads.get("perception", {})
        multimodal = perception_payload.get("multimodal_interpretations", perception_payload)
        if not isinstance(multimodal, dict):
            multimodal = {}

        conversation_window = request.get("conversation_window")
        if not isinstance(conversation_window, list):
            conversation_window = []

        completed = [name for name in self._PROVIDER_ORDER if name in pending.provider_payloads]
        missing = [name for name in self._PROVIDER_ORDER if name not in pending.provider_payloads]
        confidence = {
            "provider_coverage": len(completed) / len(self._PROVIDER_ORDER),
            "completed_providers": completed,
            "missing_providers": missing,
            "window_ms": int(self._wait_window_seconds * 1000),
            "elapsed_ms": int(elapsed * 1000),
            "partial": bool(missing),
        }

        return ContextAssembledPayload(
            input_id=input_id,
            user_input=str(request.get("user_input", "")),
            conversation_window=conversation_window,
            retrieved_facts=[str(f) for f in facts],
            social_signals=social_signals,
            multimodal_interpretations=multimodal,
            confidence=confidence,
            user_id=request.get("user_id"),
            author_id=request.get("author_id"),
            author_name=request.get("author_name"),
            channel_id=request.get("channel_id"),
            channel_context=request.get("channel_context"),
            recent_turn_summary=request.get("recent_turn_summary"),
            timestamp=request.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        )

    async def _await_and_publish(self, input_id: str) -> None:
        started = asyncio.get_running_loop().time()
        deadline = started + self._wait_window_seconds
        while asyncio.get_running_loop().time() < deadline:
            async with self._lock:
                pending = self._pending.get(input_id)
                if pending is None or pending.published:
                    return
                if len(pending.provider_payloads) == len(self._PROVIDER_ORDER):
                    break
            await asyncio.sleep(0.005)

        async with self._lock:
            pending = self._pending.get(input_id)
            if pending is None or pending.published:
                return
            pending.published = True
            elapsed = asyncio.get_running_loop().time() - started
            payload = self._build_context_payload(input_id, pending, elapsed)
            del self._pending[input_id]

        await self._publisher.publish(EventSubjects.CONTEXT_ASSEMBLED, payload, use_jetstream=True)

    async def start(self, durable_name: str = "context_assembler") -> bool:
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.INPUT_RECEIVED,
                handler=self._handle_input_received,
                use_jetstream=True,
                durable=f"{durable_name}_input",
            )
            await self._subscriber.subscribe(
                subject=EventSubjects.MEMORY_RETRIEVED,
                handler=lambda msg: self._handle_provider_response(msg, "memory"),
                use_jetstream=True,
                durable=f"{durable_name}_memory",
            )
            await self._subscriber.subscribe(
                subject=EventSubjects.SOCIAL_SIGNALS_RETRIEVED,
                handler=lambda msg: self._handle_provider_response(msg, "social"),
                use_jetstream=True,
                durable=f"{durable_name}_social",
            )
            await self._subscriber.subscribe(
                subject=EventSubjects.PERCEPTION_INTERPRET_RETRIEVED,
                handler=lambda msg: self._handle_provider_response(msg, "perception"),
                use_jetstream=True,
                durable=f"{durable_name}_perception",
            )
            return True
        except nats.errors.Error:
            logger.exception("Failed to start ContextAssemblerService")
            return False

    async def stop(self) -> None:
        await self._subscriber.unsubscribe_all()
