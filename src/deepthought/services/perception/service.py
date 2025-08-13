from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from ...eda.events import EventSubjects, PerceptionEmbeddingsPayload
from ...eda.publisher import Publisher

Worker = Callable[..., Any]
Fuser = Callable[[Sequence[Any]], dict]


@dataclass
class PerceptionService:
    """Orchestrate perception workers and publish fused embeddings.

    Parameters
    ----------
    workers:
        Sequence of callables that produce intermediate perception outputs. Each
        worker can be synchronous or asynchronous and is invoked with the same
        positional and keyword arguments provided to :meth:`run`.
    fuser:
        Callable that merges the list of worker outputs into a dictionary with
        ``spans``, ``embeddings``, ``encoders`` and ``provenance`` entries.
    publisher:
        Event publisher used to emit :class:`~deepthought.eda.events.PerceptionEmbeddingsPayload`.
    """

    workers: Sequence[Worker]
    fuser: Fuser
    publisher: Publisher

    async def run(self, message_id: str, user_id: str, *args: Any, **kwargs: Any) -> dict:
        """Execute workers, fuse their outputs and publish the result.

        Parameters
        ----------
        message_id:
            Identifier of the message being processed.
        user_id:
            Identifier of the user that produced the message.
        *args, **kwargs:
            Additional arguments forwarded to each worker.
        """

        results: list[Any] = []
        for worker in self.workers:
            result = worker(*args, **kwargs)
            if isinstance(result, Awaitable):
                result = await result
            results.append(result)

        fused = self.fuser(results)
        payload = PerceptionEmbeddingsPayload(
            message_id=message_id,
            user_id=user_id,
            spans=fused.get("spans", []),
            embeddings=fused.get("embeddings", []),
            encoders=fused.get("encoders", []),
            provenance=fused.get("provenance", {}),
        )
        await self.publisher.publish(EventSubjects.PERCEPTION_EMBEDDINGS, payload)
        return fused


async def run(*args: Any, **kwargs: Any) -> None:
    """Entry point for ``dtrt perception run``.

    Parameters
    ----------
    service:
        Optional :class:`PerceptionService` instance. If provided, the service's
        :meth:`run` method is invoked with any additional ``args`` and
        ``kwargs``. When omitted, the function returns immediately. This loose
        contract allows the CLI to expose the entry point without imposing a
        specific wiring of workers or publisher.
    """

    service: PerceptionService | None = kwargs.pop("service", None)
    if service is not None:
        await service.run(*args, **kwargs)
