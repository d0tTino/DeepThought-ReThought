from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Sequence

from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

from ...eda.events import (
    EncoderMetadata,
    EventSubjects,
    ModalityEmbeddings,
    PerceptionEmbeddingsEvent,
    PerceptionEmbeddingsPayload,
)
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
        fused: Sequence[Sequence[float]] | None = None,
        by_modality: Mapping[str, Mapping[str, Any]] | None = None,
        provenance: Dict[str, Any] | None = None,
        retries: int = 3,
    ) -> Dict | None:
        """Publish a :class:`PerceptionEmbeddingsEvent` with retries.

        Parameters
        ----------
        message_id:
            Identifier of the message being processed.
        user_id:
            Identifier of the user who produced the message.
        fused:
            Sequence of fused embedding vectors aligned to the common hop grid.
        by_modality:
            Mapping of modality name to its embeddings, spans and encoders.
        provenance:
            Provenance information about how the embeddings were generated.
        retries:
            Number of times to retry publishing on failure.
        """

        modality_payloads = {
            name: ModalityEmbeddings(
                spans=[[int(span[0]), int(span[1])] for span in meta.get("spans", [])],
                embeddings=[list(map(float, emb)) for emb in meta.get("embeddings", [])],
                encoders=[EncoderMetadata(**enc) for enc in meta.get("encoders", [])],
            )
            for name, meta in (by_modality or {}).items()
        }

        payload = PerceptionEmbeddingsPayload(
            message_id=message_id,
            user_id=user_id,
            fused=[[float(x) for x in emb] for emb in fused] if fused is not None else None,
            by_modality=modality_payloads,
        )

        # Deduplicate encoders across modalities to avoid redundant metadata
        top_encoders_map: dict[tuple[str, str | None], EncoderMetadata] = {}
        for mod in modality_payloads.values():
            for enc in mod.encoders:
                key = (enc.name, enc.modality)
                top_encoders_map.setdefault(key, enc)
        top_encoders = list(top_encoders_map.values())

        event = PerceptionEmbeddingsEvent(
            encoders=top_encoders,
            provenance=dict(provenance or {}),
            payload=payload,
        )

        return await self._publisher.publish(
            EventSubjects.PERCEPTION_EMBEDDINGS,
            event,
            use_jetstream=True,
            retries=retries,
        )
