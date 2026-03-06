from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.contracts import EventEnvelope, decode_payload_or_envelope
from ..eda.events import ContextAssembledPayload, EventSubjects
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


@dataclass
class _PendingAssembly:
    request: dict[str, Any]
    provider_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    trace_id: str | None = None
    required_providers: set[str] = field(default_factory=set)
    provider_deadlines: dict[str, float] = field(default_factory=dict)
    provider_received_at: dict[str, float] = field(default_factory=dict)
    started_at: float = 0.0
    published_reason: str | None = None
    state: "_AssemblyState" = field(default_factory=lambda: _AssemblyState.OPEN)
    published: bool = False
    event: asyncio.Event = field(default_factory=asyncio.Event)


class _AssemblyState(str, Enum):
    OPEN = "OPEN"
    PARTIAL_READY = "PARTIAL_READY"
    COMPLETE = "COMPLETE"
    TIMEOUT_PUBLISHED = "TIMEOUT_PUBLISHED"


class ContextAssemblerService:
    """Assemble contextual data from memory/social/perception providers."""

    _PROVIDER_ORDER = ("memory", "social", "perception")
    _MULTIMODAL_SCHEMA_VERSION = "multimodal.semantic-notes.v1"

    @classmethod
    def _normalize_multimodal(cls, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {
                "schema_version": cls._MULTIMODAL_SCHEMA_VERSION,
                "summary": "no multimodal signals",
                "notes": [],
                "by_modality": {},
                "attachments": None,
                "confidence": {"aggregate": 0.0, "low_confidence": True, "threshold": 0.45},
                "fallback": {"ask_clarifying_question": True, "reason": "missing multimodal interpretations"},
            }

        normalized = dict(raw)
        normalized.setdefault("schema_version", cls._MULTIMODAL_SCHEMA_VERSION)
        normalized.setdefault("summary", "no multimodal signals")
        normalized.setdefault("notes", [])
        normalized.setdefault("by_modality", {})
        normalized.setdefault("attachments", None)
        normalized.setdefault("confidence", {"aggregate": 0.0, "low_confidence": True, "threshold": 0.45})
        normalized.setdefault("fallback", {"ask_clarifying_question": False, "reason": ""})
        return normalized

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

    async def _publish_fanout_requests(self, payload: dict[str, Any], *, trace_id: str | None, causation_id: str | None) -> None:
        fanout_payload = {
            "input_id": payload.get("input_id"),
            "trace_id": trace_id,
            "user_input": payload.get("user_input"),
            "user_id": payload.get("user_id"),
            "author_id": payload.get("author_id"),
            "channel_id": payload.get("channel_id"),
            "author_name": payload.get("author_name"),
            "timestamp": payload.get("timestamp"),
            "attachments": payload.get("attachments"),
        }
        for subject in (
            EventSubjects.MEMORY_RETRIEVAL_REQUESTED,
            EventSubjects.SOCIAL_SIGNALS_REQUESTED,
            EventSubjects.PERCEPTION_INTERPRET_REQUESTED,
        ):
            envelope = EventEnvelope.build(
                subject=subject,
                payload=fanout_payload,
                producer=self.__class__.__name__,
                trace_id=trace_id,
                causation_id=causation_id,
            )
            await self._publisher.publish(subject, envelope.__dict__, use_jetstream=True)

    async def _handle_input_received(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("Input payload must be an object")
            payload, envelope_meta = decode_payload_or_envelope(EventSubjects.INPUT_RECEIVED, data)
            input_id = payload.get("input_id")
            user_input = payload.get("user_input")
            if not isinstance(input_id, str) or not isinstance(user_input, str):
                raise ValueError("Input payload missing required fields")

            loop_time = asyncio.get_running_loop().time()
            provider_deadlines = {
                provider: loop_time + self._wait_window_seconds for provider in self._PROVIDER_ORDER
            }
            pending = _PendingAssembly(
                request=payload,
                trace_id=envelope_meta.get("trace_id") if isinstance(envelope_meta.get("trace_id"), str) else None,
                required_providers=set(self._PROVIDER_ORDER),
                provider_deadlines=provider_deadlines,
                started_at=loop_time,
            )
            async with self._lock:
                self._pending[input_id] = pending

            causation_id = envelope_meta.get("event_id") if isinstance(envelope_meta.get("event_id"), str) else input_id
            await self._publish_fanout_requests(payload, trace_id=pending.trace_id, causation_id=causation_id)
            asyncio.create_task(self._await_and_publish(input_id))
            await msg.ack()
        except Exception:
            logger.exception("Failed to process input event in ContextAssemblerService")
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def _handle_provider_response(self, msg: Msg, provider_name: str) -> None:
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("Provider payload must be an object")
            provider_subject = {
                "memory": EventSubjects.MEMORY_RETRIEVED,
                "social": EventSubjects.SOCIAL_SIGNALS_RETRIEVED,
                "perception": EventSubjects.PERCEPTION_INTERPRET_RETRIEVED,
            }.get(provider_name, EventSubjects.CONTEXT_ASSEMBLED)
            payload, envelope_meta = decode_payload_or_envelope(provider_subject, data)
            input_id = payload.get("input_id")
            if not isinstance(input_id, str):
                raise ValueError("Provider payload missing input_id")
            response_trace_id = envelope_meta.get("trace_id") or payload.get("trace_id")
            if response_trace_id is not None and not isinstance(response_trace_id, str):
                raise ValueError("Provider payload trace_id must be a string when provided")

            async with self._lock:
                pending = self._pending.get(input_id)
                if pending is not None and not pending.published:
                    if pending.trace_id and response_trace_id and pending.trace_id != response_trace_id:
                        logger.debug(
                            "Ignoring %s provider response due to trace mismatch input_id=%s expected=%s actual=%s",
                            provider_name,
                            input_id,
                            pending.trace_id,
                            response_trace_id,
                        )
                        await msg.ack()
                        return
                    pending.provider_payloads[provider_name] = payload
                    pending.provider_received_at[provider_name] = asyncio.get_running_loop().time()
                    if len(pending.provider_payloads) == len(pending.required_providers):
                        pending.state = _AssemblyState.COMPLETE
                    else:
                        pending.state = _AssemblyState.PARTIAL_READY
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
        multimodal = self._normalize_multimodal(perception_payload.get("multimodal_interpretations", perception_payload))

        conversation_window = request.get("conversation_window")
        if not isinstance(conversation_window, list):
            conversation_window = []

        completed = [name for name in self._PROVIDER_ORDER if name in pending.provider_payloads]
        missing = [name for name in self._PROVIDER_ORDER if name not in pending.provider_payloads]
        provider_timings = {}
        for provider in self._PROVIDER_ORDER:
            deadline_offset_ms = int((pending.provider_deadlines[provider] - pending.started_at) * 1000)
            received_at = pending.provider_received_at.get(provider)
            received_offset_ms = int((received_at - pending.started_at) * 1000) if received_at is not None else None
            provider_timings[provider] = {
                "deadline_offset_ms": deadline_offset_ms,
                "received_offset_ms": received_offset_ms,
                "timed_out": provider in missing,
            }

        confidence = {
            "provider_coverage": len(completed) / len(self._PROVIDER_ORDER),
            "completed_providers": completed,
            "missing_providers": missing,
            "window_ms": int(self._wait_window_seconds * 1000),
            "elapsed_ms": int(elapsed * 1000),
            "partial": bool(missing),
            "publish_reason": pending.published_reason,
            "assembly_state": pending.state.value,
            "correlation": {"input_id": input_id, "trace_id": pending.trace_id},
            "provider_timings": provider_timings,
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
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._wait_window_seconds
        publish_reason = "timeout_partial"
        while loop.time() < deadline:
            async with self._lock:
                pending = self._pending.get(input_id)
                if pending is None or pending.published:
                    return
                if len(pending.provider_payloads) == len(self._PROVIDER_ORDER):
                    pending.state = _AssemblyState.COMPLETE
                    publish_reason = "all_providers_received"
                    break
            await asyncio.sleep(0.005)

        async with self._lock:
            pending = self._pending.get(input_id)
            if pending is None or pending.published:
                return
            if publish_reason == "timeout_partial":
                pending.state = _AssemblyState.TIMEOUT_PUBLISHED
            pending.published_reason = publish_reason
            pending.published = True
            elapsed = loop.time() - pending.started_at
            payload = self._build_context_payload(input_id, pending, elapsed)
            del self._pending[input_id]

        envelope = EventEnvelope.build(
            subject=EventSubjects.CONTEXT_ASSEMBLED,
            payload=json.loads(payload.to_json()),
            producer=self.__class__.__name__,
            trace_id=pending.trace_id,
            causation_id=input_id,
        )
        await self._publisher.publish(EventSubjects.CONTEXT_ASSEMBLED, envelope.__dict__, use_jetstream=True)

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
