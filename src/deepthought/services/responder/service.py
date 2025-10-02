"""Create response candidates from fused perception and memory."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, MutableMapping

from ...bus import Publisher, Subscriber
from ...eda.events import EventSubjects
from ..base import BaseService

logger = logging.getLogger(__name__)


class ResponderService(BaseService):
    """Generate candidate responses using fused perception and memory context."""

    FUSED_DURABLE = "resp_fused_v1"
    MEMORY_DURABLE = "resp_memory_v1"

    def __init__(self, subscriber: Subscriber, publisher: Publisher) -> None:
        super().__init__(subscriber, publisher)
        self._memory: MutableMapping[str, Dict[str, Any]] = {}
        self._fused: MutableMapping[str, Dict[str, Any]] = {}
        self._start_lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            await self._subscriber.subscribe(
                subject=EventSubjects.PERCEPTION_FUSED,
                handler=self._handle_fused,
                use_jetstream=True,
                durable=self.FUSED_DURABLE,
            )
            await self._subscriber.subscribe(
                subject=EventSubjects.MEMORY_RETRIEVED,
                handler=self._handle_memory,
                use_jetstream=True,
                durable=self.MEMORY_DURABLE,
            )
            self._started = True
            logger.info(
                "ResponderService listening for fused perception (%s) and memory (%s)",
                EventSubjects.PERCEPTION_FUSED,
                EventSubjects.MEMORY_RETRIEVED,
            )

    async def _handle_memory(self, msg: Any) -> None:
        payload = self._decode_message(msg)
        input_id = payload.get("input_id") or payload.get("memory_id")
        if not input_id:
            logger.warning("Memory payload missing input_id: %s", payload)
            await self._ack_message(msg)
            return
        self._memory[input_id] = payload
        logger.debug("Stored memory context for %s", input_id)
        await self._ack_message(msg)

    async def _handle_fused(self, msg: Any) -> None:
        payload = self._decode_message(msg)
        input_id = payload.get("input_id")
        if not input_id:
            logger.warning("Fused perception missing input_id: %s", payload)
            await self._ack_message(msg)
            return
        self._fused[input_id] = payload
        logger.debug("Stored fused features for %s", input_id)

        memory = self._memory.get(input_id, {})
        candidates = self._build_candidates(input_id, payload, memory)
        response_payload = {
            "input_id": input_id,
            "candidates": candidates,
            "meta": {"source": "responder_service"},
        }
        await self._publish(EventSubjects.RESPONSE_CANDIDATES, response_payload)
        await self._ack_message(msg)

    def _build_candidates(
        self,
        input_id: str,
        fused: Dict[str, Any],
        memory: Dict[str, Any],
    ) -> Any:
        features = fused.get("features", {})
        audio = features.get("audio") or []
        image = features.get("image") or []
        memory_text = memory.get("retrieved_knowledge", {}).get("summary") if isinstance(memory.get("retrieved_knowledge"), dict) else None

        candidate_primary = {
            "id": f"{input_id}-primary",
            "text": memory_text or "Default response based on perception.",
            "confidence": 0.8 if memory_text else 0.5,
        }
        candidate_secondary = {
            "id": f"{input_id}-alt",
            "text": "Alternative generated response.",
            "confidence": 0.4 + 0.1 * bool(audio) + 0.1 * bool(image),
        }
        return [candidate_primary, candidate_secondary]
