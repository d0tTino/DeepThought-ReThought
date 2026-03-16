from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Optional

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.contracts import decode_payload_or_envelope
from ..eda.events import (
    CorrectionSignalPayload,
    DiscordFeedbackSignalPayload,
    EventSubjects,
    OutcomeSignalPayload,
    ResponseRankedPayload,
)
from ..metrics.prometheus import (
    ADAPTATION_EFFECT_DELTA,
    RESPONSE_FEEDBACK_SIGNALS_TOTAL,
    RESPONSE_QUALITY_SCORE,
)
from .base import BaseService
from .db_manager import DBManager

logger = logging.getLogger(__name__)


class FeedbackService(BaseService):
    """Consume response feedback and adapt social/memory confidence state."""

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        db_manager: Optional[DBManager] = None,
        *,
        nats_url: str | None = None,
        connect_retries: int = 1,
        connect_timeout: float = 2.0,
        min_confidence: float = 0.35,
        abuse_window_seconds: float = 60.0,
        abuse_max_events: int = 12,
        export_interval_seconds: float = 60.0,
        export_min_score: float = 0.7,
    ) -> None:
        super().__init__(
            nats_client,
            js_context,
            nats_url=nats_url,
            connect_retries=connect_retries,
            connect_timeout=connect_timeout,
        )
        self._db = db_manager or DBManager()
        self._recent_responses: dict[str, dict[str, Any]] = {}
        self._min_confidence = min_confidence
        self._abuse_window_seconds = max(5.0, abuse_window_seconds)
        self._abuse_max_events = max(1, abuse_max_events)
        self._export_interval_seconds = max(5.0, export_interval_seconds)
        self._export_min_score = export_min_score
        self._feedback_activity: dict[str, deque[float]] = {}
        self._export_task: asyncio.Task[None] | None = None

    async def _handle_response_ranked(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            payload_data, _meta = decode_payload_or_envelope(EventSubjects.RESPONSE_RANKED, data)
            payload = ResponseRankedPayload.from_dict(payload_data)
            if payload.input_id:
                self._recent_responses[payload.input_id] = {
                    "user_id": payload.user_id,
                    "author_id": payload.author_id,
                    "source": payload.source,
                    "model_id": None,
                    "confidence": payload.confidence,
                    "final_response": payload.final_response,
                    "timestamp": payload.timestamp,
                }
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:
            logger.error("Failed to process response ranked feedback context", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def _handle_outcome_signal(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            payload_data, _meta = decode_payload_or_envelope(EventSubjects.OUTCOME_SIGNAL, data)
            payload = OutcomeSignalPayload.from_dict(payload_data)
            await self._apply_feedback(
                signal_type="outcome",
                signal=payload.signal.lower(),
                input_id=payload.input_id,
                user_id=payload.user_id or payload.author_id,
                source=payload.response_source,
                model_id=None,
                affinity_delta=payload.affinity_delta,
                confidence_delta=payload.confidence_delta,
                signal_confidence=1.0,
                details={"timestamp": payload.timestamp},
            )
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:
            logger.error("Failed to process outcome signal", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def _handle_correction_signal(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            payload_data, _meta = decode_payload_or_envelope(EventSubjects.CORRECTION_SIGNAL, data)
            payload = CorrectionSignalPayload.from_dict(payload_data)
            await self._apply_feedback(
                signal_type="correction",
                signal="corrected",
                input_id=payload.input_id,
                user_id=payload.user_id or payload.author_id,
                source=payload.response_source,
                model_id=None,
                affinity_delta=payload.affinity_delta,
                confidence_delta=payload.confidence_delta,
                signal_confidence=1.0,
                details={
                    "correction": payload.correction,
                    "prior_response": payload.prior_response,
                    "timestamp": payload.timestamp,
                },
            )
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:
            logger.error("Failed to process correction signal", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def _handle_discord_feedback_signal(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            payload_data, _meta = decode_payload_or_envelope(EventSubjects.DISCORD_FEEDBACK_SIGNAL, data)
            payload = DiscordFeedbackSignalPayload.from_dict(payload_data)
            affinity_delta, confidence_delta = self._discord_feedback_to_deltas(payload)
            await self._apply_feedback(
                signal_type=payload.signal_type,
                signal=payload.signal,
                input_id=payload.input_id,
                user_id=payload.user_id or payload.author_id,
                source=payload.response_source,
                model_id=payload.model_id,
                affinity_delta=affinity_delta,
                confidence_delta=confidence_delta,
                signal_confidence=payload.confidence,
                details={
                    "message_id": payload.message_id,
                    "metadata": payload.metadata,
                    "timestamp": payload.timestamp,
                },
            )
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:
            logger.error("Failed to process discord feedback signal", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    def _discord_feedback_to_deltas(self, payload: DiscordFeedbackSignalPayload) -> tuple[float, float]:
        if payload.signal_type == "reaction":
            if payload.signal == "positive":
                return 0.75, 0.08
            return -0.6, -0.08
        if payload.signal_type == "message_edit":
            return -0.25, -0.04
        if payload.signal_type == "explicit_correction":
            return -0.4, -0.12
        return 0.0, 0.0

    def _is_throttled(self, actor_id: str | None) -> bool:
        if not actor_id:
            return False
        now = monotonic()
        history = self._feedback_activity.setdefault(actor_id, deque())
        while history and now - history[0] > self._abuse_window_seconds:
            history.popleft()
        history.append(now)
        return len(history) > self._abuse_max_events

    async def _apply_feedback(
        self,
        *,
        signal_type: str,
        signal: str,
        input_id: str | None,
        user_id: str | None,
        source: str | None,
        model_id: str | None,
        affinity_delta: float,
        confidence_delta: float,
        signal_confidence: float,
        details: dict[str, Any],
    ) -> None:
        source_label = source or "unknown"
        model_label = model_id or "unknown"
        resolved_user = user_id
        if input_id and input_id in self._recent_responses:
            cached = self._recent_responses[input_id]
            resolved_user = resolved_user or cached.get("author_id") or cached.get("user_id")
            source_label = source_label if source else str(cached.get("source") or "unknown")
            model_label = model_label if model_id else str(cached.get("model_id") or "unknown")

        if signal_type == "outcome" and affinity_delta == 0:
            affinity_delta = 1.0 if signal in {"positive", "success", "thumbs_up"} else -1.0
        if signal_type == "outcome" and confidence_delta == 0:
            confidence_delta = 0.1 if signal in {"positive", "success", "thumbs_up"} else -0.1

        score = min(1.0, abs(affinity_delta) * 0.35 + abs(confidence_delta) * 2.5 + float(signal_confidence) * 0.4)
        if float(signal_confidence) < self._min_confidence:
            details["discard_reason"] = "low_confidence"
            return
        if self._is_throttled(resolved_user):
            details["discard_reason"] = "throttled"
            return

        if resolved_user:
            await self._db.adjust_affinity(resolved_user, affinity_delta)
            await self._db.adjust_theory_confidence(resolved_user, confidence_delta)

        await self._db.record_feedback_signal(
            signal_type=signal_type,
            signal=signal,
            input_id=input_id,
            user_id=resolved_user,
            source=source_label,
            model_id=model_label,
            confidence=signal_confidence,
            score=score,
            details=details,
        )

        RESPONSE_FEEDBACK_SIGNALS_TOTAL.labels(
            signal_type=signal_type,
            signal=signal,
            source=source_label,
        ).inc()
        RESPONSE_QUALITY_SCORE.labels(source=source_label).observe(1.0 if confidence_delta > 0 else 0.0)
        ADAPTATION_EFFECT_DELTA.labels(target="affinity").observe(abs(float(affinity_delta)))
        ADAPTATION_EFFECT_DELTA.labels(target="memory_confidence").observe(abs(float(confidence_delta)))

        telemetry_payload = {
            "event": "response_feedback",
            "signal_type": signal_type,
            "signal": signal,
            "input_id": input_id,
            "user_id": resolved_user,
            "source": source_label,
            "model_id": model_label,
            "signal_confidence": signal_confidence,
            "feedback_score": score,
            "affinity_delta": affinity_delta,
            "confidence_delta": confidence_delta,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._publisher.publish(
            "dtr.telemetry.response_feedback.v1",
            telemetry_payload,
            use_jetstream=True,
        )

    async def _run_export_loop(self) -> None:
        while True:
            await asyncio.sleep(self._export_interval_seconds)
            if not self._publisher:
                continue
            tuples = await self._db.fetch_high_value_feedback(limit=100, min_score=self._export_min_score)
            if not tuples:
                continue
            await self._publisher.publish(
                "dtr.training.feedback_tuples.v1",
                {"count": len(tuples), "items": tuples, "exported_at": datetime.now(timezone.utc).isoformat()},
                use_jetstream=True,
            )

    async def start(self, durable_name: str = "feedback_service") -> bool:
        self._subscriptions.clear()
        self.add_subscription(
            subject=EventSubjects.RESPONSE_RANKED,
            handler=self._handle_response_ranked,
            use_jetstream=True,
            durable=f"{durable_name}_ranked",
        )
        self.add_subscription(
            subject=EventSubjects.OUTCOME_SIGNAL,
            handler=self._handle_outcome_signal,
            use_jetstream=True,
            durable=f"{durable_name}_outcome",
        )
        self.add_subscription(
            subject=EventSubjects.CORRECTION_SIGNAL,
            handler=self._handle_correction_signal,
            use_jetstream=True,
            durable=f"{durable_name}_correction",
        )
        self.add_subscription(
            subject=EventSubjects.DISCORD_FEEDBACK_SIGNAL,
            handler=self._handle_discord_feedback_signal,
            use_jetstream=True,
            durable=f"{durable_name}_discord",
        )
        started = await super().start()
        if started:
            self._export_task = asyncio.create_task(self._run_export_loop())
        return started

    async def stop(self) -> None:
        if self._export_task is not None:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
            self._export_task = None
        await super().stop()
