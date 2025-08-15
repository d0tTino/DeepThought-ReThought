from __future__ import annotations

import asyncio
from typing import Dict, Sequence, Tuple

import numpy as np

from deepthought.services.perception.service import PerceptionService


class DummyPublisher:
    def __init__(self) -> None:
        self.kwargs: Dict | None = None

    async def publish(self, **kwargs) -> None:  # pragma: no cover - simple async stub
        self.kwargs = kwargs


class DummyTextWorker:
    def __call__(self, tokens: Sequence[Tuple[str, float, float]], memmap_path: str) -> np.memmap:
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        return mm


def dummy_fuser(modalities):  # pragma: no cover - used only in tests
    return next(iter(modalities.values()))


def test_service_fuses_and_publishes():
    publisher = DummyPublisher()
    service = PerceptionService(publisher, text_worker=DummyTextWorker(), fuser=dummy_fuser)

    asyncio.run(
        service.run(
            message_id="m1",
            user_id="u1",
            text_tokens=[("hi", 0.0, 0.1)],
        )
    )

    assert publisher.kwargs is not None
    assert publisher.kwargs["message_id"] == "m1"
    assert publisher.kwargs["user_id"] == "u1"
    assert publisher.kwargs["embeddings"] == [[2.0, 3.0]]
