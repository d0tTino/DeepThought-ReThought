from __future__ import annotations

import asyncio
from typing import Dict, Sequence, Tuple

import numpy as np

from deepthought.metrics.prometheus import INPUT_LATENCY_SECONDS, INPUTS_TOTAL
from deepthought.services.perception.service import PerceptionService


class DummyPublisher:
    def __init__(self) -> None:
        self.kwargs: Dict | None = None

    async def publish(self, **kwargs) -> None:  # pragma: no cover - simple async stub
        self.kwargs = kwargs


class DummyTextWorker:
    def __call__(self, tokens: Sequence[Tuple[str, float, float]], memmap_path: str):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        times = np.array([[0.0, 0.05], [0.05, 0.1]], dtype=np.float32)
        return mm, times


def _counter_value(counter, service: str) -> float:
    for sample in counter.collect()[0].samples:
        if sample.labels.get("service") == service:
            return sample.value
    return 0.0


def _hist_count(hist, service: str) -> float:
    for sample in hist.collect()[0].samples:
        if sample.name.endswith("_count") and sample.labels.get("service") == service:
            return sample.value
    return 0.0


def test_spans_and_metrics_increment() -> None:
    publisher = DummyPublisher()
    service = PerceptionService(publisher, text_worker=DummyTextWorker())

    total_before = _counter_value(INPUTS_TOTAL, "perception_service")
    count_before = _hist_count(INPUT_LATENCY_SECONDS, "perception_service")

    asyncio.run(
        service.run(
            message_id="m1",
            user_id="u1",
            text_tokens=[("hi", 0.0, 0.1)],
        )
    )

    assert publisher.kwargs is not None
    spans = publisher.kwargs["by_modality"]["text"]["spans"]
    assert spans == [[0, 50], [50, 100]]
    assert publisher.kwargs["spans"] == spans
    assert publisher.kwargs["modality_mask"]["text"] == [True, True]

    total_after = _counter_value(INPUTS_TOTAL, "perception_service")
    count_after = _hist_count(INPUT_LATENCY_SECONDS, "perception_service")

    assert total_after == total_before + 1
    assert count_after == count_before + 1
