"""Fuse perception embeddings into a multimodal representation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, MutableMapping

from ...bus import Publisher, Subscriber
from ...eda.events import EventSubjects
from ..base import BaseService

logger = logging.getLogger(__name__)


class PerceptionService(BaseService):
    """Fuse audio and image embeddings and publish combined features."""

    AUDIO_DURABLE = "perc_audio_v1"
    IMAGE_DURABLE = "perc_image_v1"

    def __init__(self, subscriber: Subscriber, publisher: Publisher) -> None:
        super().__init__(subscriber, publisher)
        self._embeddings: MutableMapping[str, Dict[str, Any]] = {}
        self._start_lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            await self._subscriber.subscribe(
                subject=EventSubjects.PERCEPTION_AUDIO_EMBED,
                handler=self._handle_audio,
                use_jetstream=True,
                durable=self.AUDIO_DURABLE,
            )
            await self._subscriber.subscribe(
                subject=EventSubjects.PERCEPTION_IMAGE_EMBED,
                handler=self._handle_image,
                use_jetstream=True,
                durable=self.IMAGE_DURABLE,
            )
            self._started = True
            logger.info(
                "PerceptionService listening for audio (%s) and image (%s) embeddings",
                EventSubjects.PERCEPTION_AUDIO_EMBED,
                EventSubjects.PERCEPTION_IMAGE_EMBED,
            )

    async def _handle_audio(self, msg: Any) -> None:
        payload = self._decode_message(msg)
        await self._update_embedding("audio", payload)
        await self._ack_message(msg)

    async def _handle_image(self, msg: Any) -> None:
        payload = self._decode_message(msg)
        await self._update_embedding("image", payload)
        await self._ack_message(msg)

    async def _update_embedding(self, modality: str, payload: Dict[str, Any]) -> None:
        input_id = payload.get("input_id") or payload.get("audio_id") or payload.get("image_id")
        if not input_id:
            logger.warning("Perception payload missing identifier: %s", payload)
            return
        record = self._embeddings.setdefault(input_id, {})
        record[modality] = payload
        logger.debug("Updated %s embedding for %s", modality, input_id)

        if "audio" in record and "image" in record:
            fused_payload = {
                "input_id": input_id,
                "features": {
                    "audio": record["audio"].get("embedding"),
                    "image": record["image"].get("embedding"),
                },
                "metadata": {
                    "audio_meta": record["audio"].get("metadata"),
                    "image_meta": record["image"].get("metadata"),
                },
                "meta": {"source": "perception_service"},
            }
            logger.debug("Publishing fused perception payload for %s", input_id)
            await self._publish(EventSubjects.PERCEPTION_FUSED, fused_payload)
