from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Sequence

from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

from ...eda.events import EventSubjects, PerceptionEmbeddingsPayload
from ...eda.publisher import Publisher

logger = logging.getLogger(__name__)


class PerceptionPublisher:
    """Publish perception embedding events via JetStream."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._publisher = Publisher(nats_client, js_context)

    async def publish(
        self,
        message_id: str,
        user_id: str,
        *,
        spans: Sequence[Sequence[int]] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        encoders: Sequence[Dict[str, Any]] | None = None,
        provenance: Dict[str, Any] | None = None,
        retries: int = 3,
    ) -> Dict | None:
        """Publish a :class:`PerceptionEmbeddingsPayload` with retries.

        Parameters
        ----------
        message_id:
            Identifier of the message being processed.
        user_id:
            Identifier of the user who produced the message.
        spans:
            Span indices for each embedding.
        embeddings:
            Vector embeddings.
        encoders:
            Metadata describing each encoder.
        provenance:
            Provenance information about how the embeddings were generated.
        retries:
            Number of times to retry publishing on failure.
        """

        payload = PerceptionEmbeddingsPayload(
            message_id=message_id,
            user_id=user_id,
            spans=list(spans or []),
            embeddings=[list(map(float, emb)) for emb in (embeddings or [])],
            encoders=[dict(meta) for meta in (encoders or [])],
            provenance=dict(provenance or {}),
        )

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return await self._publisher.publish(
                    EventSubjects.PERCEPTION_EMBEDDINGS,
                    payload,
                    use_jetstream=True,
                )
            except Exception as err:  # pragma: no cover - network issues
                last_error = err
                logger.warning("Publish attempt %s failed for message %s: %s", attempt, message_id, err)
                await asyncio.sleep(min(0.1 * attempt, 1.0))
        assert last_error is not None
        logger.error("Failed to publish perception embeddings after %s attempts", retries)
        raise last_error
