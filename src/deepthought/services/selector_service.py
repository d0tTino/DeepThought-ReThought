from __future__ import annotations

import json
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.contracts import EventEnvelope, decode_payload_or_envelope
from ..eda.events import (
    EventSubjects,
    ResponseCandidate,
    ResponseCandidatesPayload,
    ResponseRankedPayload,
)
from ..eda.publisher import Publisher, publish_enveloped
from ..eda.subscriber import Subscriber
from . import moderation
from .policy_engine import VersionedPolicyEngine

logger = logging.getLogger(__name__)


@dataclass
class _AggregationState:
    input_id: str
    candidates: list[ResponseCandidate] = field(default_factory=list)
    user_id: Optional[str] = None
    author_id: Optional[str] = None
    channel_id: Optional[str] = None
    interaction_policy: Optional[dict] = None
    context_confidence: dict = field(default_factory=dict)
    social_intent_hints: dict = field(default_factory=dict)
    user_history_affinity: dict[str, float] = field(default_factory=dict)
    adaptation_state: dict = field(default_factory=dict)
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    flush_task: asyncio.Task[None] | None = None


class SelectorService:
    """Select the best candidate response and publish a ranked result."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        safety_filter: Optional[Callable[[ResponseCandidate], bool]] = None,
        window_seconds: float = 0.15,
        early_exit_confidence: float = 0.9,
        source_confidence_weights: Optional[dict[str, float]] = None,
        source_calibration_profiles: Optional[dict[str, dict[str, float]]] = None,
        toxicity_guard: Optional[Callable[[ResponseCandidate], bool]] = None,
        contradiction_checker: Optional[Callable[[ResponseCandidate, list[ResponseCandidate]], bool]] = None,
        repetition_penalty: Optional[Callable[[ResponseCandidate, list[ResponseCandidate]], float]] = None,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._safety_filter = safety_filter
        self._window_seconds = max(0.0, window_seconds)
        self._early_exit_confidence = early_exit_confidence
        self._source_confidence_weights = source_confidence_weights or {}
        self._source_calibration_profiles = {
            key.lower(): value
            for key, value in (source_calibration_profiles or {}).items()
            if isinstance(value, dict)
        }
        self._toxicity_guard = toxicity_guard or self._default_toxicity_guard
        self._contradiction_checker = contradiction_checker or (lambda _c, _all: True)
        self._repetition_penalty = repetition_penalty or self._default_repetition_penalty
        self._pending_by_input: dict[str, _AggregationState] = {}
        self._aggregation_lock = asyncio.Lock()
        self._context_degradation_weight = 0.25
        self._policy_fit_weight = 0.2
        self._history_affinity_weight = 0.15
        self._policy_engine = VersionedPolicyEngine()

    def _default_toxicity_guard(self, candidate: ResponseCandidate) -> bool:
        score, _ = moderation.evaluate_toxicity(candidate.text)
        return score < moderation.TOXICITY_THRESHOLD

    def _default_repetition_penalty(
        self,
        candidate: ResponseCandidate,
        all_candidates: list[ResponseCandidate],
    ) -> float:
        duplicates = sum(1 for item in all_candidates if item.text.strip().lower() == candidate.text.strip().lower())
        return min(0.3, max(0, duplicates - 1) * 0.1)

    def _source_adaptation(self, adaptation_state: Optional[dict], source: str) -> dict:
        if not isinstance(adaptation_state, dict):
            return {}
        sources = adaptation_state.get("sources")
        if not isinstance(sources, dict):
            return {}
        profile = sources.get(source)
        return dict(profile) if isinstance(profile, dict) else {}

    def _normalized_confidence(self, candidate: ResponseCandidate, adaptation_state: Optional[dict] = None) -> float:
        source = (candidate.source or "default").lower()
        weight = self._source_confidence_weights.get(source, self._source_confidence_weights.get("default", 1.0))
        source_adaptation = self._source_adaptation(adaptation_state, source)
        selector_profile = source_adaptation.get("selector") if isinstance(source_adaptation.get("selector"), dict) else {}
        if isinstance(selector_profile.get("weight_multiplier"), (int, float)):
            weight *= float(selector_profile["weight_multiplier"])
        calibrated = max(0.0, candidate.confidence * weight)
        profile = self._source_calibration_profiles.get(source) or self._source_calibration_profiles.get("default") or {}
        adaptation_confidence = source_adaptation.get("confidence") if isinstance(source_adaptation.get("confidence"), dict) else {}
        adaptation_calibration = adaptation_confidence.get("calibration") if isinstance(adaptation_confidence, dict) else {}
        if isinstance(adaptation_calibration, dict):
            profile = {**profile, **adaptation_calibration}
        slope = float(profile.get("slope", 1.0))
        bias = float(profile.get("bias", 0.0))
        floor = float(profile.get("floor", 0.0))
        ceiling = float(profile.get("ceiling", 1.0))

        metadata = candidate.source_metadata if isinstance(candidate.source_metadata, dict) else {}
        if isinstance(metadata.get("calibration"), dict):
            source_calibration = metadata["calibration"]
            slope *= float(source_calibration.get("slope", 1.0))
            bias += float(source_calibration.get("bias", 0.0))

        calibrated = slope * calibrated + bias
        return max(floor, min(ceiling, calibrated))

    def _deterministic_tiebreak_key(self, candidate: ResponseCandidate) -> tuple[str, str, str]:
        source = (candidate.source or "").strip().lower()
        text = (candidate.text or "").strip().lower()
        rationale = ",".join(sorted(candidate.rationale_tags)) if isinstance(candidate.rationale_tags, list) else ""
        return (source, text, rationale)

    def _choose_fallback(self, interaction_policy: Optional[dict], adaptation_state: Optional[dict] = None) -> tuple[str, str]:
        policy = interaction_policy or {}
        fallback_profile = adaptation_state.get("fallback") if isinstance(adaptation_state, dict) and isinstance(adaptation_state.get("fallback"), dict) else {}
        aggressiveness = fallback_profile.get("aggressiveness", policy.get("fallback_aggressiveness", 0.5))
        try:
            fallback_aggressiveness = max(0.0, min(1.0, float(aggressiveness)))
        except (TypeError, ValueError):
            fallback_aggressiveness = 0.5
        should_clarify = policy.get("ask_clarifying_on_no_safe", True) and fallback_aggressiveness <= 0.6
        if should_clarify:
            return (
                "I don't yet have a safe, reliable answer. Could you clarify what outcome you want?",
                "clarifying_question",
            )
        return ("I want to help, but I need a safer way to answer this request.", "safe_default")

    def _context_degradation_score(self, context_confidence: Optional[dict]) -> float:
        context = context_confidence or {}
        aggregate_raw = context.get("aggregate", 1.0)
        threshold_raw = context.get("threshold", 0.45)
        low_confidence = bool(context.get("low_confidence"))
        try:
            aggregate = max(0.0, min(1.0, float(aggregate_raw)))
        except (TypeError, ValueError):
            aggregate = 1.0
        try:
            threshold = max(0.0, min(1.0, float(threshold_raw)))
        except (TypeError, ValueError):
            threshold = 0.45

        degradation = max(0.0, threshold - aggregate)
        if low_confidence:
            degradation += 0.1
        return max(0.0, min(1.0, degradation))

    def _policy_fit_score(
        self,
        candidate: ResponseCandidate,
        interaction_policy: Optional[dict],
        social_intent_hints: Optional[dict],
    ) -> float:
        policy = interaction_policy or {}
        hints = social_intent_hints or {}
        score = 0.0

        if bool(candidate.safety_passed) or candidate.safety_passed is None:
            score += 0.3
        if policy.get("ask_clarifying_on_no_safe") and bool(hints.get("clarify_preferred")):
            score += 0.2

        expected_style = policy.get("response_style") or hints.get("preferred_style")
        candidate_style = candidate.safety_metadata.get("style") if isinstance(candidate.safety_metadata, dict) else None
        if isinstance(expected_style, str) and isinstance(candidate_style, str):
            if expected_style.strip().lower() == candidate_style.strip().lower():
                score += 0.5
        elif expected_style is None:
            score += 0.2

        return max(0.0, min(1.0, score))

    def _user_history_affinity_score(
        self,
        candidate: ResponseCandidate,
        user_history_affinity: Optional[dict[str, float]],
        social_intent_hints: Optional[dict],
    ) -> float:
        affinity = user_history_affinity or {}
        hints = social_intent_hints or {}
        source_key = (candidate.source or "default").lower()
        source_affinity = affinity.get(source_key, affinity.get("default", 0.0))
        intent_affinity = affinity.get("intent", 0.0)
        if bool(hints.get("high_rapport_expected")):
            intent_affinity += 0.1
        raw_score = 0.7 * source_affinity + 0.3 * intent_affinity
        return max(-1.0, min(1.0, float(raw_score)))

    def _rank_candidates(
        self,
        candidates: list[ResponseCandidate],
        interaction_policy: Optional[dict] = None,
        context_confidence: Optional[dict] = None,
        social_intent_hints: Optional[dict] = None,
        user_history_affinity: Optional[dict[str, float]] = None,
        adaptation_state: Optional[dict] = None,
    ) -> tuple[list[ResponseCandidate], list[dict]]:
        diagnostics: list[dict] = []
        scored: list[tuple[float, ResponseCandidate]] = []
        context_degradation = self._context_degradation_score(context_confidence)
        context_adjustment = -self._context_degradation_weight * context_degradation

        for candidate in candidates:
            rejection_reasons: list[str] = []
            normalized = self._normalized_confidence(candidate, adaptation_state)
            policy_fit = self._policy_fit_score(candidate, interaction_policy, social_intent_hints)
            policy_adjustment = self._policy_fit_weight * policy_fit
            affinity_score = self._user_history_affinity_score(candidate, user_history_affinity, social_intent_hints)
            affinity_adjustment = self._history_affinity_weight * affinity_score

            if not (self._safety_filter(candidate) if self._safety_filter else candidate.safety_passed is not False):
                rejection_reasons.append("safety_filter")
            if not self._toxicity_guard(candidate):
                rejection_reasons.append("toxicity")
            if not self._contradiction_checker(candidate, candidates):
                rejection_reasons.append("contradiction")
            penalty = self._repetition_penalty(candidate, candidates)
            final_score = max(
                0.0,
                normalized + policy_adjustment + affinity_adjustment + context_adjustment - penalty + min(0.05, 0.01 * len(candidate.rationale_tags or [])),
            )
            policy_decision = self._policy_engine.evaluate_candidate(
                text=candidate.text,
                confidence=normalized,
                prior_artifacts=self._candidate_policy_artifacts(candidate),
            )
            if not policy_decision.allowed:
                rejection_reasons.append(f"policy:{policy_decision.action}")
            factor_scores = {
                "base_confidence": normalized,
                "context_degradation": context_degradation,
                "context_adjustment": context_adjustment,
                "policy_fit": policy_fit,
                "policy_adjustment": policy_adjustment,
                "history_affinity": affinity_score,
                "history_adjustment": affinity_adjustment,
                "repetition_penalty": penalty,
                "policy_risk_level": policy_decision.risk_level,
                "policy_confidence_band": policy_decision.confidence_band,
            }
            source_adaptation = self._source_adaptation(adaptation_state, (candidate.source or "default").lower())
            if source_adaptation:
                factor_scores["adaptation"] = source_adaptation
            if rejection_reasons:
                diagnostics.append(
                    {
                        "text": candidate.text,
                        "source": candidate.source,
                        "confidence": candidate.confidence,
                        "normalized_confidence": normalized,
                        "score": final_score,
                        "factor_scores": factor_scores,
                        "rejection_reasons": rejection_reasons,
                        "policy_artifacts": [*self._candidate_policy_artifacts(candidate), policy_decision.artifacts],
                    }
                )
                continue
            scored.append((final_score, candidate))
            diagnostics.append(
                {
                    "text": candidate.text,
                    "source": candidate.source,
                    "confidence": candidate.confidence,
                    "normalized_confidence": normalized,
                    "score": final_score,
                    "factor_scores": factor_scores,
                    "rejection_reasons": [],
                    "policy_artifacts": [*self._candidate_policy_artifacts(candidate), policy_decision.artifacts],
                }
            )

        ranked = [
            candidate
            for _, candidate in sorted(
                scored,
                key=lambda item: (-item[0], self._deterministic_tiebreak_key(item[1])),
            )
        ]
        return ranked, sorted(diagnostics, key=lambda item: item["score"], reverse=True)

    def _candidate_policy_artifacts(self, candidate: ResponseCandidate) -> list[dict]:
        metadata = candidate.safety_metadata if isinstance(candidate.safety_metadata, dict) else {}
        artifacts = metadata.get("policy_artifacts")
        if isinstance(artifacts, list):
            return [item for item in artifacts if isinstance(item, dict)]
        return []

    async def _publish_selection(self, state: _AggregationState, trace_id: str | None = None, causation_id: str | None = None) -> None:
        ranked_candidates, diagnostics = self._rank_candidates(
            state.candidates,
            interaction_policy={**(state.interaction_policy or {}), "policy_version": self._policy_engine.VERSION},
            context_confidence=state.context_confidence,
            social_intent_hints=state.social_intent_hints,
            user_history_affinity=state.user_history_affinity,
            adaptation_state=state.adaptation_state,
        )
        fallback_reason = None
        if ranked_candidates:
            selected = ranked_candidates[0]
            final_response = selected.text
            selected_diag = next(
                (
                    item
                    for item in diagnostics
                    if item.get("text") == selected.text and item.get("source") == selected.source
                ),
                None,
            )
            final_confidence = float(selected_diag.get("score", self._normalized_confidence(selected))) if selected_diag else self._normalized_confidence(selected)
            final_source = selected.source
            selected_policy_artifacts = self._candidate_policy_artifacts(selected)
        else:
            final_response, fallback_reason = self._choose_fallback(state.interaction_policy, state.adaptation_state)
            final_confidence = 0.0
            final_source = "selector_fallback"
            selected_policy_artifacts = []

        ranked_payload = ResponseRankedPayload(
            final_response=final_response,
            input_id=state.input_id,
            user_id=state.author_id or state.user_id,
            author_id=state.author_id,
            channel_id=state.channel_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=final_confidence,
            source=final_source,
            interaction_policy={**(state.interaction_policy or {}), "policy_version": self._policy_engine.VERSION},
            candidates=ranked_candidates,
        )
        envelope = EventEnvelope.build(
            subject=EventSubjects.RESPONSE_RANKED,
            payload=json.loads(ranked_payload.to_json()),
            producer=self.__class__.__name__,
            trace_id=trace_id,
            causation_id=causation_id or state.input_id,
        )
        await self._publisher.publish(
            EventSubjects.RESPONSE_RANKED,
            envelope.__dict__,
            use_jetstream=True,
            timeout=10.0,
        )

        telemetry_payload = {
            "input_id": state.input_id,
            "chosen_source": final_source,
            "selected_confidence": final_confidence,
            "fallback_reason": fallback_reason,
            "chosen_policy_artifacts": selected_policy_artifacts,
            "policy_version": self._policy_engine.VERSION,
            "window_seconds": self._window_seconds,
            "candidate_count": len(state.candidates),
            "context_confidence": state.context_confidence,
            "social_intent_hints": state.social_intent_hints,
            "user_history_affinity": state.user_history_affinity,
            "adaptation_state": state.adaptation_state,
            "weights": {
                "context_degradation": self._context_degradation_weight,
                "policy_fit": self._policy_fit_weight,
                "history_affinity": self._history_affinity_weight,
            },
            "diagnostics": diagnostics,
        }
        await publish_enveloped(
            self._publisher,
            subject="dtr.telemetry.selector_ranking.v1",
            payload=telemetry_payload,
            producer=self.__class__.__name__,
            trace_id=trace_id,
            causation_id=causation_id,
            use_jetstream=True,
            timeout=10.0,
        )

    async def _flush_input_id(self, input_id: str, trace_id: str | None = None, causation_id: str | None = None) -> None:
        async with self._aggregation_lock:
            state = self._pending_by_input.pop(input_id, None)
        if state is None:
            return
        await self._publish_selection(state, trace_id=trace_id, causation_id=causation_id)

    async def _schedule_flush(self, input_id: str, trace_id: str | None = None, causation_id: str | None = None) -> None:
        await asyncio.sleep(self._window_seconds)
        await self._flush_input_id(input_id, trace_id=trace_id, causation_id=causation_id)

    async def _handle_candidates_event(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("ResponseCandidates payload must be a dict")
            decoded_payload, envelope_meta = decode_payload_or_envelope(EventSubjects.RESPONSE_CANDIDATES, data)
            payload = ResponseCandidatesPayload.from_dict(decoded_payload)
            trace_id = envelope_meta.get("trace_id") if isinstance(envelope_meta.get("trace_id"), str) else None
            causation_id = envelope_meta.get("event_id") if isinstance(envelope_meta.get("event_id"), str) else None
            input_id = payload.input_id or "global"

            async with self._aggregation_lock:
                state = self._pending_by_input.get(input_id)
                if state is None:
                    state = _AggregationState(
                        input_id=input_id,
                        user_id=payload.user_id,
                        author_id=payload.author_id,
                        channel_id=payload.channel_id,
                        interaction_policy=payload.interaction_policy,
                        context_confidence=payload.context_confidence or {},
                        social_intent_hints=payload.social_intent_hints or {},
                        user_history_affinity=payload.user_history_affinity or {},
                        adaptation_state=payload.adaptation_state or {},
                    )
                    self._pending_by_input[input_id] = state

                state.candidates.extend(payload.candidates)
                state.user_id = state.user_id or payload.user_id
                state.author_id = state.author_id or payload.author_id
                state.channel_id = state.channel_id or payload.channel_id
                state.interaction_policy = state.interaction_policy or payload.interaction_policy
                state.context_confidence = state.context_confidence or (payload.context_confidence or {})
                state.social_intent_hints = state.social_intent_hints or (payload.social_intent_hints or {})
                state.user_history_affinity = state.user_history_affinity or (payload.user_history_affinity or {})
                state.adaptation_state = state.adaptation_state or (payload.adaptation_state or {})

                ranked_candidates, diagnostics = self._rank_candidates(
                    state.candidates,
                    interaction_policy={**(state.interaction_policy or {}), "policy_version": self._policy_engine.VERSION},
                    context_confidence=state.context_confidence,
                    social_intent_hints=state.social_intent_hints,
                    user_history_affinity=state.user_history_affinity,
                    adaptation_state=state.adaptation_state,
                )
                top_score = diagnostics[0]["score"] if diagnostics else 0.0
                early_exit = bool(ranked_candidates) and top_score >= self._early_exit_confidence
                if early_exit:
                    if state.flush_task and not state.flush_task.done():
                        state.flush_task.cancel()
                    flush_now = True
                else:
                    flush_now = False
                    if state.flush_task is None or state.flush_task.done():
                        state.flush_task = asyncio.create_task(self._schedule_flush(input_id, trace_id=trace_id, causation_id=causation_id))

            await msg.ack()
            if flush_now:
                await self._flush_input_id(input_id, trace_id=trace_id, causation_id=causation_id)
        except (json.JSONDecodeError, ValueError):
            logger.error("Invalid response candidates payload", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()
            elif hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:
            logger.error("SelectorService failed to process candidates", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def start(self, durable_name: str = "selector_service") -> bool:
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.RESPONSE_CANDIDATES,
                handler=self._handle_candidates_event,
                use_jetstream=True,
                durable=durable_name,
            )
            return True
        except nats.errors.Error:
            logger.error("SelectorService failed to subscribe", exc_info=True)
            return False

    async def stop(self) -> None:
        async with self._aggregation_lock:
            flush_tasks = [state.flush_task for state in self._pending_by_input.values() if state.flush_task and not state.flush_task.done()]
            self._pending_by_input.clear()
        for task in flush_tasks:
            task.cancel()
        await self._subscriber.unsubscribe_all()
