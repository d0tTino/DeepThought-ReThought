"""Minimal perception service leveraging a NATS publisher stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from .publisher import PerceptionPublisher


@dataclass
class PerceptionService:
    """Thin wrapper around :class:`PerceptionPublisher`.

    Parameters
    ----------
    publisher:
        Event publisher used to emit
        :class:`~deepthought.eda.events.PerceptionEmbeddingsPayload`.
    """

    publisher: PerceptionPublisher

    async def run(
        self,
        message_id: str,
        user_id: str,
        *,
        spans: Sequence[Sequence[int]] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        encoders: Sequence[Dict[str, Any]] | None = None,
        provenance: Dict[str, Any] | None = None,
    ) -> None:
        """Publish a perception payload."""

        await self.publisher.publish(
            message_id=message_id,
            user_id=user_id,
            spans=spans,
            embeddings=embeddings,
            encoders=encoders,
            provenance=provenance,
        )


async def run(*args: Any, **kwargs: Any) -> None:
    """Entry point for ``dtrt perception run``."""

    service: PerceptionService | None = kwargs.pop("service", None)
    if service is not None:
        await service.run(*args, **kwargs)
