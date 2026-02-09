from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)


class SelectorService:
    """Select the best candidate response and publish a ranked result."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        safety_filter: Optional[Callable[[ResponseCandidate], bool]] = None,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._safety_filter = safety_filter

    def _rank_candidates(self, candidates: list[ResponseCandidate]) -> list[ResponseCandidate]:
        filtered = [c for c in candidates if (self._safety_filter(c) if self._safety_filter else c.safety_passed is not False)]
        return sorted(filtered, key=lambda c: c.confidence, reverse=True)

    async def _handle_candidates_event(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("ResponseCandidates payload must be a dict")
            payload = ResponseCandidatesPayload.from_dict(data)
            ranked_candidates = self._rank_candidates(payload.candidates)
            if not ranked_candidates:
                await msg.ack()
                logger.warning("SelectorService received empty candidates for input_id=%s", payload.input_id)
                return

            selected = ranked_candidates[0]
            ranked_payload = ResponseRankedPayload(
                final_response=selected.text,
                input_id=payload.input_id,
                user_id=payload.user_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                confidence=selected.confidence,
                source=selected.source,
                candidates=ranked_candidates,
            )
            await self._publisher.publish(
                EventSubjects.RESPONSE_RANKED,
                ranked_payload,
                use_jetstream=True,
                timeout=10.0,
            )
            await msg.ack()
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
        await self._subscriber.unsubscribe_all()
