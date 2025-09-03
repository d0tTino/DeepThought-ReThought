import asyncio
import sys
from typing import Dict, Sequence, Tuple

import asyncio
import sys
import types
from pathlib import Path
import importlib.util

import numpy as np
import pytest

torch = sys.modules.get("torch")
if not getattr(torch, "nn", None):  # pragma: no cover - optional dependency missing
    pytest.skip("torch not available", allow_module_level=True)

# Bypass heavy package imports by injecting lightweight modules
root = Path(__file__).resolve().parents[2] / "src"
services_pkg = types.ModuleType("deepthought.services")
services_pkg.__path__ = [str(root / "deepthought" / "services")]
sys.modules.setdefault("deepthought.services", services_pkg)
modules_pkg = types.ModuleType("deepthought.modules")
modules_pkg.__path__ = [str(root / "deepthought" / "modules")]
sys.modules.setdefault("deepthought.modules", modules_pkg)

# Load ModalityFuser directly and expose through the lightweight package
spec = importlib.util.spec_from_file_location(
    "deepthought.modules.fuser", root / "deepthought" / "modules" / "fuser.py"
)
fuser_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fuser_module)
sys.modules["deepthought.modules.fuser"] = fuser_module
modules_pkg.ModalityFuser = fuser_module.ModalityFuser

# Stub out video worker to avoid OpenCV dependency
wv_module = types.ModuleType("deepthought.services.perception.worker_video")
class _DummyVideoWorker:  # pragma: no cover - simple stub
    pass

wv_module.VideoPerceptionWorker = _DummyVideoWorker
sys.modules.setdefault("deepthought.services.perception.worker_video", wv_module)

from deepthought.metrics.prometheus import MISSING_MODALITY_TOTAL
from deepthought.modules.fuser import ModalityFuser
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


def _counter_value(counter, modality: str) -> float:
    for sample in counter.collect()[0].samples:
        if sample.labels.get("modality") == modality:
            return sample.value
    return 0.0


def test_missing_modality_triggers_metrics(caplog):
    publisher = DummyPublisher()
    fuser = ModalityFuser({"text": 2, "audio": 2}, fused_dim=3)
    fuser.eval()
    service = PerceptionService(publisher, text_worker=DummyTextWorker(), fuser=fuser)

    before = _counter_value(MISSING_MODALITY_TOTAL, "audio")
    with caplog.at_level("WARNING"):
        asyncio.run(
            service.run(
                message_id="m1",
                user_id="u1",
                text_tokens=[("hi", 0.0, 0.1)],
            )
        )
    after = _counter_value(MISSING_MODALITY_TOTAL, "audio")

    assert after == before + 1
    assert "audio modality absent" in caplog.text
    assert publisher.kwargs is not None
    fused = np.asarray(publisher.kwargs["fused"], dtype=float)
    assert fused.shape == (2, 3)
