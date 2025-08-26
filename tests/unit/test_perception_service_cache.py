from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
deepthought_stub = types.ModuleType("deepthought")
deepthought_stub.__path__ = [str(ROOT / "src/deepthought")]
services_stub = types.ModuleType("deepthought.services")
services_stub.__path__ = [str(ROOT / "src/deepthought/services")]
perception_stub = types.ModuleType("deepthought.services.perception")
perception_stub.__path__ = [str(ROOT / "src/deepthought/services/perception")]
sys.modules.setdefault("deepthought", deepthought_stub)
sys.modules.setdefault("deepthought.services", services_stub)
sys.modules.setdefault("deepthought.services.perception", perception_stub)
service_mod = importlib.import_module("deepthought.services.perception.service")
PerceptionService = service_mod.PerceptionService


class DummyPublisher:
    async def publish(self, **kwargs) -> None:  # pragma: no cover - simple async stub
        pass


class CountingTextWorker:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, tokens: Sequence[Tuple[str, float, float]], memmap_path: str):
        self.calls += 1
        data = np.array([[1.0, 2.0]], dtype="float32")
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        times = np.array([[0.0, 0.03]], dtype=np.float32)
        return mm, times


def test_service_reuses_text_cache(tmp_path):
    os.environ["DT_PERCEPTION_TEXT_CACHE_DIR"] = str(tmp_path)
    try:
        publisher = DummyPublisher()
        worker = CountingTextWorker()
        service = PerceptionService(publisher, text_worker=worker)
        tokens = [("hi", 0.0, 0.03)]
        for _ in range(2):
            asyncio.run(
                service.run(
                    message_id="m1",
                    user_id="u1",
                    text_tokens=tokens,
                )
            )
        assert worker.calls == 1
        meta_files = list(tmp_path.glob("*_meta.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert "timestamps" in meta and "encoder" in meta
    finally:
        del os.environ["DT_PERCEPTION_TEXT_CACHE_DIR"]
