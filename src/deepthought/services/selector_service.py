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

from ..eda.events import (
    EventSubjects,
    ResponseCandidate,
    ResponseCandidatesPayload,
    ResponseRankedPayload,
)
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from . import moderation

logger = logging.getLogger(__name__)


@dataclass
class _AggregationState:
    input_id: str
    candidates: list[ResponseCandidate] = field(default_factory=list)
    user_id: Optional[str] = None
    author_id: Optional[str] = None
    channel_id: Optional[str] = None
    interaction_policy: Optional[dict] = None
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
        self._toxicity_guard = toxicity_guard or self._default_toxicity_guard
        self._contradiction_checker = contradiction_checker or (lambda _c, _all: True)
        self._repetition_penalty = repetition_penalty or self._default_repetition_penalty
        self._pending_by_input: dict[str, _AggregationState] = {}
        self._aggregation_lock = asyncio.Lock()

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

    def _normalized_confidence(self, candidate: ResponseCandidate) -> float:
        source = (candidate.source or "default").lower()
        weight = self._source_confidence_weights.get(source, self._source_confidence_weights.get("default", 1.0))
        return max(0.0, candidate.confidence * weight)

    def _choose_fallback(self, interaction_policy: Optional[dict]) -> tuple[str, str]:
        policy = interaction_policy or {}
        if policy.get("ask_clarifying_on_no_safe", True):
            return (
                "I don't yet have a safe, reliable answer. Could you clarify what outcome you want?",
                "clarifying_question",
            )
        return ("I want to help, but I need a safer way to answer this request.", "safe_default")

    def _rank_candidates(self, candidates: list[ResponseCandidate]) -> tuple[list[ResponseCandidate], list[dict]]:
        diagnostics: list[dict] = []
        scored: list[tuple[float, ResponseCandidate]] = []
        for candidate in candidates:
            rejection_reasons: list[str] = []
            normalized = self._normalized_confidence(candidate)
            if not (self._safety_filter(candidate) if self._safety_filter else candidate.safety_passed is not False):
                rejection_reasons.append("safety_filter")
            if not self._toxicity_guard(candidate):
                rejection_reasons.append("toxicity")
            if not self._contradiction_checker(candidate, candidates):
                rejection_reasons.append("contradiction")
            penalty = self._repetition_penalty(candidate, candidates)
            final_score = max(0.0, normalized - penalty)
            if rejection_reasons:
                diagnostics.append(
                    {
                        "text": candidate.text,
                        "source": candidate.source,
                        "confidence": candidate.confidence,
                        "normalized_confidence": normalized,
                        "score": final_score,
                        "rejection_reasons": rejection_reasons,
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
                    "rejection_reasons": [],
                }
            )

        ranked = [candidate for _, candidate in sorted(scored, key=lambda item: item[0], reverse=True)]
        return ranked, sorted(diagnostics, key=lambda item: item["score"], reverse=True)

    async def _publish_selection(self, state: _AggregationState) -> None:
        ranked_candidates, diagnostics = self._rank_candidates(state.candidates)
        fallback_reason = None
        if ranked_candidates:
            selected = ranked_candidates[0]
            final_response = selected.text
            final_confidence = self._normalized_confidence(selected)
            final_source = selected.source
        else:
            final_response, fallback_reason = self._choose_fallback(state.interaction_policy)
            final_confidence = 0.0
            final_source = "selector_fallback"

        ranked_payload = ResponseRankedPayload(
            final_response=final_response,
            input_id=state.input_id,
            user_id=state.author_id or state.user_id,
            author_id=state.author_id,
            channel_id=state.channel_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=final_confidence,
            source=final_source,
            interaction_policy=state.interaction_policy,
            candidates=ranked_candidates,
        )
        await self._publisher.publish(
            EventSubjects.RESPONSE_RANKED,
            ranked_payload,
            use_jetstream=True,
            timeout=10.0,
        )

        telemetry_payload = {
            "input_id": state.input_id,
            "chosen_source": final_source,
            "selected_confidence": final_confidence,
            "fallback_reason": fallback_reason,
            "window_seconds": self._window_seconds,
            "candidate_count": len(state.candidates),
            "diagnostics": diagnostics,
        }
        await self._publisher.publish(
            "dtr.telemetry.selector_ranking.v1",
            telemetry_payload,
            use_jetstream=True,
            timeout=10.0,
        )

    async def _flush_input_id(self, input_id: str) -> None:
        async with self._aggregation_lock:
            state = self._pending_by_input.pop(input_id, None)
        if state is None:
            return
        await self._publish_selection(state)

    async def _schedule_flush(self, input_id: str) -> None:
        await asyncio.sleep(self._window_seconds)
        await self._flush_input_id(input_id)

    async def _handle_candidates_event(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("ResponseCandidates payload must be a dict")
            payload = ResponseCandidatesPayload.from_dict(data)
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
                    )
                    self._pending_by_input[input_id] = state

                state.candidates.extend(payload.candidates)
                state.user_id = state.user_id or payload.user_id
                state.author_id = state.author_id or payload.author_id
                state.channel_id = state.channel_id or payload.channel_id
                state.interaction_policy = state.interaction_policy or payload.interaction_policy

                ranked_candidates, _ = self._rank_candidates(state.candidates)
                early_exit = bool(ranked_candidates) and self._normalized_confidence(ranked_candidates[0]) >= self._early_exit_confidence
                if early_exit:
                    if state.flush_task and not state.flush_task.done():
                        state.flush_task.cancel()
                    flush_now = True
                else:
                    flush_now = False
                    if state.flush_task is None or state.flush_task.done():
                        state.flush_task = asyncio.create_task(self._schedule_flush(input_id))

            await msg.ack()
            if flush_now:
                await self._flush_input_id(input_id)
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
