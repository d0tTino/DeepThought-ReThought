"""Rank responder candidates."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from ...bus import Publisher, Subscriber
from ...eda.events import EventSubjects
from ..base import BaseService

logger = logging.getLogger(__name__)


class SelectorService(BaseService):
    """Rank response candidates and publish the selected result."""

    CANDIDATE_DURABLE = "selector_candidates_v1"

    def __init__(self, subscriber: Subscriber, publisher: Publisher) -> None:
        super().__init__(subscriber, publisher)
        self._start_lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            await self._subscriber.subscribe(
                subject=EventSubjects.RESPONSE_CANDIDATES,
                handler=self._handle_candidates,
                use_jetstream=True,
                durable=self.CANDIDATE_DURABLE,
            )
            self._started = True
            logger.info("SelectorService listening on %s", EventSubjects.RESPONSE_CANDIDATES)

    async def _handle_candidates(self, msg: Any) -> None:
        payload = self._decode_message(msg)
        candidates: List[Dict[str, Any]] = payload.get("candidates") or []
        if not isinstance(candidates, list):
            logger.warning("Invalid candidates payload: %s", payload)
            await self._ack_message(msg)
            return

        ranked = sorted(
            candidates,
            key=lambda candidate: candidate.get("confidence", 0.0),
            reverse=True,
        )
        ranked_payload = {
            "input_id": payload.get("input_id"),
            "ranked_candidates": ranked,
            "selected_index": 0 if ranked else None,
            "meta": {"source": "selector_service"},
        }
        await self._publish(EventSubjects.RESPONSE_RANKED, ranked_payload)
        await self._ack_message(msg)
