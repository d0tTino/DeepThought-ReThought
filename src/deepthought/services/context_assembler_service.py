from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.contracts import EventEnvelope, decode_payload_or_envelope
from ..eda.events import ContextAssembledPayload, EventSubjects
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
if TYPE_CHECKING:  # pragma: no cover
    from .db_manager import DBManager

logger = logging.getLogger(__name__)


class _NullAdaptationStore:
    async def get_adaptation_state(self, *, user_id: str | None = None) -> dict[str, Any]:
        return {}


@dataclass
class _PendingAssembly:
    request: dict[str, Any]
    provider_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    trace_id: str | None = None
    required_providers: set[str] = field(default_factory=set)
    provider_deadlines: dict[str, float] = field(default_factory=dict)
    provider_received_at: dict[str, float] = field(default_factory=dict)
    provider_missing_reasons: dict[str, str | None] = field(default_factory=dict)
    started_at: float = 0.0
    published_reason: str | None = None
    state: "_AssemblyState" = field(default_factory=lambda: _AssemblyState.OPEN)
    published: bool = False
    event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _PublishedAssembly:
    payload: dict[str, Any]
    trace_id: str | None
    expires_at: float
    missing_at_publish: set[str]


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
        db_manager: "DBManager | None" = None,
        *,
        wait_window_seconds: float = 0.2,
        provider_jitter_budget_seconds: float = 0.01,
        min_provider_timeout_seconds: float = 0.01,
        max_provider_timeout_seconds: float = 0.75,
        latency_history_size: int = 64,
        late_arrival_window_seconds: float = 0.1,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        if db_manager is None:
            try:
                from .db_manager import DBManager

                self._db = DBManager()
            except ModuleNotFoundError:
                self._db = _NullAdaptationStore()
        else:
            self._db = db_manager
        self._wait_window_seconds = max(0.01, wait_window_seconds)
        self._provider_jitter_budget_seconds = max(0.0, provider_jitter_budget_seconds)
        self._min_provider_timeout_seconds = max(0.005, min_provider_timeout_seconds)
        self._max_provider_timeout_seconds = max(self._min_provider_timeout_seconds, max_provider_timeout_seconds)
        self._latency_history_size = max(5, latency_history_size)
        self._late_arrival_window_seconds = max(0.0, late_arrival_window_seconds)

        self._provider_latency_history: dict[str, deque[float]] = {
            provider: deque(maxlen=self._latency_history_size) for provider in self._PROVIDER_ORDER
        }
        self._pending: dict[str, _PendingAssembly] = {}
        self._recently_published: dict[str, _PublishedAssembly] = {}
        self._lock = asyncio.Lock()

    async def _load_adaptation_state(self, request: dict[str, Any]) -> dict[str, Any]:
        principal = request.get("author_id") or request.get("user_id")
        if not isinstance(principal, str) or not principal.strip():
            return {}
        try:
            return await self._db.get_adaptation_state(user_id=principal)
        except Exception:
            logger.exception("Failed to load adaptation state for principal=%s", principal)
            return {}

    @staticmethod
    def _merge_adaptation_into_social_signals(
        social_signals: dict[str, Any],
        adaptation_state: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(social_signals)
        selector_inputs = dict(merged.get("selector_inputs") or {})
        user_profile = adaptation_state.get("user") if isinstance(adaptation_state.get("user"), dict) else {}
        response_style = None
        if isinstance(user_profile.get("response_style"), dict):
            response_style = user_profile["response_style"].get("preferred")
        fallback_profile = adaptation_state.get("fallback") if isinstance(adaptation_state.get("fallback"), dict) else {}
        source_profiles = adaptation_state.get("sources") if isinstance(adaptation_state.get("sources"), dict) else {}

        interaction_policy = dict(selector_inputs.get("interaction_policy") or {})
        if isinstance(response_style, str) and response_style.strip():
            interaction_policy["response_style"] = response_style.strip().lower()
        if "aggressiveness" in fallback_profile:
            interaction_policy["fallback_aggressiveness"] = fallback_profile.get("aggressiveness")

        user_history_affinity = dict(selector_inputs.get("user_history_affinity") or {})
        for source_name, source_profile in source_profiles.items():
            if not isinstance(source_profile, dict):
                continue
            selector_profile = source_profile.get("selector")
            if not isinstance(selector_profile, dict):
                continue
            weight_multiplier = selector_profile.get("weight_multiplier")
            if isinstance(weight_multiplier, (int, float)):
                user_history_affinity.setdefault(str(source_name), max(-1.0, min(1.0, float(weight_multiplier) - 1.0)))

        selector_inputs["interaction_policy"] = interaction_policy
        selector_inputs["user_history_affinity"] = user_history_affinity
        selector_inputs["adaptation_state"] = adaptation_state
        merged["selector_inputs"] = selector_inputs
        merged["adaptation_state"] = adaptation_state
        return merged

    @staticmethod
    def _estimate_p95(latencies: deque[float]) -> float | None:
        if not latencies:
            return None
        ordered = sorted(latencies)
        index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.95)))
        return ordered[index]

    def _provider_timeout_seconds(self, provider: str) -> float:
        p95 = self._estimate_p95(self._provider_latency_history[provider])
        baseline = p95 if p95 is not None else self._wait_window_seconds
        return max(
            self._min_provider_timeout_seconds,
            min(self._max_provider_timeout_seconds, baseline + self._provider_jitter_budget_seconds),
        )

    @staticmethod
    def _provider_has_data(provider_name: str, payload: dict[str, Any]) -> bool:
        if provider_name == "memory":
            retrieved = payload.get("retrieved_knowledge", {})
            facts = retrieved.get("facts") if isinstance(retrieved, dict) else None
            return bool(facts) if isinstance(facts, list) else bool(retrieved)
        if provider_name == "social":
            social = payload.get("social_signals", payload)
            return isinstance(social, dict) and bool(social)
        if provider_name == "perception":
            raw = payload.get("multimodal_interpretations", payload)
            if not isinstance(raw, dict):
                return False
            return bool(raw.get("notes") or raw.get("by_modality") or raw.get("summary"))
        return bool(payload)

    @staticmethod
    def _provider_explicit_unavailable(payload: dict[str, Any]) -> bool:
        status = payload.get("status")
        return bool(payload.get("unavailable") is True or status == "unavailable")

    def _derive_provider_missing_reason(
        self,
        provider: str,
        pending: _PendingAssembly,
        *,
        now: float,
    ) -> str | None:
        if provider in pending.provider_missing_reasons and pending.provider_missing_reasons[provider]:
            return pending.provider_missing_reasons[provider]
        if provider not in pending.provider_payloads:
            return "timeout" if now >= pending.provider_deadlines[provider] else None

        payload = pending.provider_payloads[provider]
        if self._provider_explicit_unavailable(payload):
            return "unavailable"
        if not self._provider_has_data(provider, payload):
            return "no_data"
        return None

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
                provider: loop_time + self._provider_timeout_seconds(provider) for provider in self._PROVIDER_ORDER
            }
            adaptation_state = await self._load_adaptation_state(payload)
            payload["adaptation_state"] = adaptation_state
            pending = _PendingAssembly(
                request=payload,
                trace_id=envelope_meta.get("trace_id") if isinstance(envelope_meta.get("trace_id"), str) else None,
                required_providers=set(self._PROVIDER_ORDER),
                provider_deadlines=provider_deadlines,
                provider_missing_reasons={provider: None for provider in self._PROVIDER_ORDER},
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

    async def _emit_context_update(self, input_id: str, entry: _PublishedAssembly, provider_name: str) -> None:
        envelope = EventEnvelope.build(
            subject=EventSubjects.CONTEXT_UPDATED,
            payload=entry.payload,
            producer=self.__class__.__name__,
            trace_id=entry.trace_id,
            causation_id=input_id,
        )
        await self._publisher.publish(EventSubjects.CONTEXT_UPDATED, envelope.__dict__, use_jetstream=True)
        logger.debug("Published %s late-arrival context update for input_id=%s", provider_name, input_id)

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

            loop_now = asyncio.get_running_loop().time()
            late_update_entry: _PublishedAssembly | None = None
            async with self._lock:
                self._purge_expired_published(loop_now)
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
                    pending.provider_received_at[provider_name] = loop_now
                    pending.provider_missing_reasons[provider_name] = self._derive_provider_missing_reason(
                        provider_name,
                        pending,
                        now=loop_now,
                    )
                    latency = max(0.0, loop_now - pending.started_at)
                    self._provider_latency_history[provider_name].append(latency)
                    pending.state = (
                        _AssemblyState.COMPLETE
                        if len(pending.provider_payloads) == len(pending.required_providers)
                        else _AssemblyState.PARTIAL_READY
                    )
                    pending.event.set()
                else:
                    recent = self._recently_published.get(input_id)
                    if (
                        recent is not None
                        and provider_name in recent.missing_at_publish
                        and recent.expires_at >= loop_now
                    ):
                        late_update_entry = recent
                        recent.payload = self._merge_late_provider_payload(recent.payload, provider_name, payload)
                        recent.payload["confidence"]["update_reason"] = "late_provider_merge"
                        recent.payload["confidence"]["late_update"] = True
                        recent.missing_at_publish.discard(provider_name)

            if late_update_entry is not None:
                await self._emit_context_update(input_id, late_update_entry, provider_name)
            await msg.ack()
        except Exception:
            logger.exception("Failed to process provider response for %s", provider_name)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    def _merge_late_provider_payload(
        self,
        assembled_payload: dict[str, Any],
        provider_name: str,
        provider_payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload = dict(assembled_payload)
        confidence = dict(payload.get("confidence") or {})
        completed = list(confidence.get("completed_providers") or [])
        missing = list(confidence.get("missing_providers") or [])
        missing_reasons = dict(confidence.get("provider_missing_reasons") or {})
        provider_timings = dict(confidence.get("provider_timings") or {})

        if provider_name == "memory":
            retrieved = provider_payload.get("retrieved_knowledge", {})
            facts = retrieved.get("facts") if isinstance(retrieved, dict) else []
            payload["retrieved_facts"] = [str(f) for f in facts] if isinstance(facts, list) else []
        elif provider_name == "social":
            social = provider_payload.get("social_signals", provider_payload)
            payload["social_signals"] = social if isinstance(social, dict) else {}
        elif provider_name == "perception":
            payload["multimodal_interpretations"] = self._normalize_multimodal(
                provider_payload.get("multimodal_interpretations", provider_payload)
            )

        if provider_name not in completed:
            completed.append(provider_name)
        missing = [item for item in missing if item != provider_name]
        missing_reasons.pop(provider_name, None)

        timing = dict(provider_timings.get(provider_name) or {})
        timing["timed_out"] = False
        timing["missing_reason"] = None
        provider_timings[provider_name] = timing

        confidence["completed_providers"] = [p for p in self._PROVIDER_ORDER if p in completed]
        confidence["missing_providers"] = [p for p in self._PROVIDER_ORDER if p in missing]
        confidence["provider_missing_reasons"] = missing_reasons
        confidence["provider_timings"] = provider_timings
        confidence["provider_coverage"] = len(confidence["completed_providers"]) / len(self._PROVIDER_ORDER)
        confidence["partial"] = bool(confidence["missing_providers"])
        payload["confidence"] = confidence
        return payload

    def _build_context_payload(self, input_id: str, pending: _PendingAssembly, elapsed: float) -> ContextAssembledPayload:
        request = pending.request
        memory_payload = pending.provider_payloads.get("memory", {})
        retrieved = memory_payload.get("retrieved_knowledge", {})
        facts = retrieved.get("facts") if isinstance(retrieved, dict) else []
        if not isinstance(facts, list):
            facts = []
        retrieval_layers = retrieved.get("layers") if isinstance(retrieved, dict) else {}
        if not isinstance(retrieval_layers, dict):
            retrieval_layers = {}

        social_payload = pending.provider_payloads.get("social", {})
        social_signals = social_payload.get("social_signals", social_payload)
        if not isinstance(social_signals, dict):
            social_signals = {}
        adaptation_state = request.get("adaptation_state") if isinstance(request.get("adaptation_state"), dict) else {}
        social_signals = self._merge_adaptation_into_social_signals(social_signals, adaptation_state)

        perception_payload = pending.provider_payloads.get("perception", {})
        multimodal = self._normalize_multimodal(perception_payload.get("multimodal_interpretations", perception_payload))

        request_window = request.get("conversation_window")
        memory_window = retrieved.get("conversation_window") if isinstance(retrieved, dict) else None
        conversation_window = []
        for candidate in (request_window, memory_window):
            if not isinstance(candidate, list):
                continue
            for turn in candidate:
                if isinstance(turn, dict):
                    conversation_window.append(turn)
        if conversation_window:
            deduped_window = []
            seen_turns: set[tuple[str, str]] = set()
            for turn in conversation_window:
                key = (str(turn.get("message_id") or ""), str(turn.get("timestamp") or ""))
                if key in seen_turns:
                    continue
                seen_turns.add(key)
                deduped_window.append(turn)
            conversation_window = deduped_window[-8:]

        recent_turn_summary = request.get("recent_turn_summary")
        if not isinstance(recent_turn_summary, str) or not recent_turn_summary.strip():
            memory_summary = retrieved.get("recent_turn_summary") if isinstance(retrieved, dict) else None
            recent_turn_summary = memory_summary if isinstance(memory_summary, str) and memory_summary.strip() else None

        retrieval_policy = retrieved.get("retrieval_policy") if isinstance(retrieved, dict) else None
        if not isinstance(retrieval_policy, dict):
            retrieval_policy = {}
        retrieval_adaptation = adaptation_state.get("retrieval") if isinstance(adaptation_state.get("retrieval"), dict) else {}
        if retrieval_adaptation:
            retrieval_policy = {**retrieval_policy, **retrieval_adaptation}

        completed = [name for name in self._PROVIDER_ORDER if name in pending.provider_payloads]
        now = pending.started_at + elapsed
        missing = []
        missing_reasons: dict[str, str] = {}
        provider_timings = {}
        for provider in self._PROVIDER_ORDER:
            deadline_offset_ms = int((pending.provider_deadlines[provider] - pending.started_at) * 1000)
            received_at = pending.provider_received_at.get(provider)
            received_offset_ms = int((received_at - pending.started_at) * 1000) if received_at is not None else None
            missing_reason = self._derive_provider_missing_reason(provider, pending, now=now)
            if missing_reason:
                missing.append(provider)
                missing_reasons[provider] = missing_reason
            provider_timings[provider] = {
                "deadline_offset_ms": deadline_offset_ms,
                "received_offset_ms": received_offset_ms,
                "timed_out": missing_reason == "timeout",
                "missing_reason": missing_reason,
            }

        provider_latency_p95_ms = {
            provider: int(self._estimate_p95(self._provider_latency_history[provider]) * 1000)
            if self._estimate_p95(self._provider_latency_history[provider]) is not None
            else None
            for provider in self._PROVIDER_ORDER
        }
        provider_timeout_budget_ms = {
            provider: int((pending.provider_deadlines[provider] - pending.started_at) * 1000)
            for provider in self._PROVIDER_ORDER
        }

        confidence = {
            "provider_coverage": len(completed) / len(self._PROVIDER_ORDER),
            "completed_providers": completed,
            "missing_providers": missing,
            "provider_missing_reasons": missing_reasons,
            "window_ms": int(self._wait_window_seconds * 1000),
            "elapsed_ms": int(elapsed * 1000),
            "partial": bool(missing),
            "publish_reason": pending.published_reason,
            "assembly_state": pending.state.value,
            "correlation": {"input_id": input_id, "trace_id": pending.trace_id},
            "provider_timings": provider_timings,
            "provider_latency_p95_ms": provider_latency_p95_ms,
            "provider_timeout_budget_ms": provider_timeout_budget_ms,
            "retrieval_layers": retrieval_layers,
            "retrieval_policy": retrieval_policy,
        }

        return ContextAssembledPayload(
            input_id=input_id,
            user_input=str(request.get("user_input", "")),
            conversation_window=conversation_window,
            retrieved_facts=[str(f) for f in facts],
            social_signals=social_signals,
            multimodal_interpretations=multimodal,
            confidence=confidence,
            adaptation_state=adaptation_state,
            user_id=request.get("user_id"),
            author_id=request.get("author_id"),
            author_name=request.get("author_name"),
            channel_id=request.get("channel_id"),
            channel_context=request.get("channel_context"),
            recent_turn_summary=recent_turn_summary,
            timestamp=request.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        )

    def _purge_expired_published(self, now: float) -> None:
        expired = [input_id for input_id, entry in self._recently_published.items() if entry.expires_at < now]
        for input_id in expired:
            del self._recently_published[input_id]

    async def _await_and_publish(self, input_id: str) -> None:
        loop = asyncio.get_running_loop()
        publish_reason = "timeout_partial"

        while True:
            async with self._lock:
                pending = self._pending.get(input_id)
                if pending is None or pending.published:
                    return

                now = loop.time()
                all_received = len(pending.provider_payloads) == len(self._PROVIDER_ORDER)
                timed_out = all(now >= pending.provider_deadlines[p] for p in self._PROVIDER_ORDER if p not in pending.provider_payloads)
                if all_received:
                    pending.state = _AssemblyState.COMPLETE
                    publish_reason = "all_providers_received"
                    break
                if timed_out:
                    pending.state = _AssemblyState.TIMEOUT_PUBLISHED
                    publish_reason = "timeout_partial"
                    break

                next_deadline = min(
                    pending.provider_deadlines[p]
                    for p in self._PROVIDER_ORDER
                    if p not in pending.provider_payloads
                )
                sleep_for = max(0.001, min(0.01, next_deadline - now))

            await asyncio.sleep(sleep_for)

        async with self._lock:
            pending = self._pending.get(input_id)
            if pending is None or pending.published:
                return

            pending.published_reason = publish_reason
            pending.published = True
            elapsed = loop.time() - pending.started_at
            payload = self._build_context_payload(input_id, pending, elapsed)
            payload_dict = json.loads(payload.to_json())
            missing = set(payload_dict.get("confidence", {}).get("missing_providers", []))
            self._recently_published[input_id] = _PublishedAssembly(
                payload=payload_dict,
                trace_id=pending.trace_id,
                expires_at=loop.time() + self._late_arrival_window_seconds,
                missing_at_publish=missing,
            )
            del self._pending[input_id]

        envelope = EventEnvelope.build(
            subject=EventSubjects.CONTEXT_ASSEMBLED,
            payload=payload_dict,
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
