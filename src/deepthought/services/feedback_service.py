from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import (
    CorrectionSignalPayload,
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

    async def _handle_response_ranked(self, msg: Msg) -> None:
        try:
            payload = ResponseRankedPayload.from_dict(json.loads(msg.data.decode()))
            if payload.input_id:
                self._recent_responses[payload.input_id] = {
                    "user_id": payload.user_id,
                    "author_id": payload.author_id,
                    "source": payload.source,
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
            payload = OutcomeSignalPayload.from_dict(json.loads(msg.data.decode()))
            await self._apply_feedback(
                signal_type="outcome",
                signal=payload.signal.lower(),
                input_id=payload.input_id,
                user_id=payload.user_id or payload.author_id,
                source=payload.response_source,
                affinity_delta=payload.affinity_delta,
                confidence_delta=payload.confidence_delta,
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
            payload = CorrectionSignalPayload.from_dict(json.loads(msg.data.decode()))
            await self._apply_feedback(
                signal_type="correction",
                signal="corrected",
                input_id=payload.input_id,
                user_id=payload.user_id or payload.author_id,
                source=payload.response_source,
                affinity_delta=payload.affinity_delta,
                confidence_delta=payload.confidence_delta,
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

    async def _apply_feedback(
        self,
        *,
        signal_type: str,
        signal: str,
        input_id: str | None,
        user_id: str | None,
        source: str | None,
        affinity_delta: float,
        confidence_delta: float,
        details: dict[str, Any],
    ) -> None:
        source_label = source or "unknown"
        resolved_user = user_id
        if input_id and input_id in self._recent_responses:
            cached = self._recent_responses[input_id]
            resolved_user = resolved_user or cached.get("author_id") or cached.get("user_id")
            source_label = source_label if source else str(cached.get("source") or "unknown")

        if signal_type == "outcome" and affinity_delta == 0:
            affinity_delta = 1.0 if signal in {"positive", "success", "thumbs_up"} else -1.0
        if signal_type == "outcome" and confidence_delta == 0:
            confidence_delta = 0.1 if signal in {"positive", "success", "thumbs_up"} else -0.1

        if resolved_user:
            await self._db.adjust_affinity(resolved_user, affinity_delta)
            await self._db.adjust_theory_confidence(resolved_user, confidence_delta)

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
        return await super().start()
