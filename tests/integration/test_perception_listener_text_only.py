"""Integration test ensuring text-only perception inputs are processed."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Iterable, Sequence
from unittest.mock import AsyncMock

import numpy as np
import pytest

import deepthought.services.perception.service as service_module
from deepthought.services.perception.listener import PerceptionServiceListener
from deepthought.services.perception.service import PerceptionService


class FakeNATS:
    """Minimal NATS client stub used by the listener."""

    is_connected = True


def _make_fake_torch() -> types.ModuleType:
    """Create a lightweight torch replacement for environments without PyTorch."""

    torch_stub = types.ModuleType("torch")

    class _Tensor:
        def __init__(self, array: Iterable[float] | np.ndarray):
            self._array = np.asarray(array, dtype=np.float32)

        def numel(self) -> int:
            return int(self._array.size)

        @property
        def ndim(self) -> int:
            return int(self._array.ndim)

        def mean(self, dim: int | None = None) -> "_Tensor":
            return _Tensor(self._array.mean(axis=dim))

        def detach(self) -> "_Tensor":
            return _Tensor(self._array.copy())

        def clone(self) -> "_Tensor":  # pragma: no cover - defensive
            return _Tensor(self._array.copy())

        def unsqueeze(self, dim: int) -> "_Tensor":  # pragma: no cover - defensive
            return _Tensor(np.expand_dims(self._array, axis=dim))

        def size(self, dim: int | None = None):  # pragma: no cover - defensive
            if dim is None:
                return self._array.shape
            return self._array.shape[dim]

        def tolist(self):
            return self._array.tolist()

        def __iter__(self):  # pragma: no cover - defensive
            for item in self._array:
                yield item

        def __array__(self, dtype=None):  # pragma: no cover - defensive
            return np.asarray(self._array, dtype=dtype)

    def _coerce(value):
        if isinstance(value, _Tensor):
            return value._array
        return np.asarray(value, dtype=np.float32)

    def from_numpy(array):
        return _Tensor(np.asarray(array, dtype=np.float32))

    def tensor(data, dtype=None):
        if dtype is not None:
            return _Tensor(np.asarray(data, dtype=np.float32))
        return _Tensor(np.asarray(data, dtype=np.float32))

    def zeros(shape, dtype=None):
        return _Tensor(np.zeros(shape, dtype=np.float32))

    def stack(seq: Sequence[_Tensor], dim: int = 0):  # pragma: no cover - defensive
        arrays = [_coerce(item) for item in seq]
        return _Tensor(np.stack(arrays, axis=dim))

    class _NoGrad:
        def __enter__(self):  # pragma: no cover - defensive
            return None

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover - defensive
            return False

    torch_stub.Tensor = _Tensor
    torch_stub.float32 = np.float32
    torch_stub.from_numpy = from_numpy
    torch_stub.tensor = tensor
    torch_stub.zeros = zeros
    torch_stub.stack = stack
    torch_stub.no_grad = lambda: _NoGrad()
    existing = getattr(service_module, "torch", None)
    symbool = getattr(existing, "SymBool", type("SymBool", (), {})) if existing else type("SymBool", (), {})
    torch_stub.SymBool = symbool
    return torch_stub


def _ensure_lightweight_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the perception service has tensor helpers when PyTorch is absent."""

    torch_mod = getattr(service_module, "torch", None)
    if torch_mod is None or not hasattr(torch_mod, "from_numpy"):
        fake = _make_fake_torch()
        monkeypatch.setitem(sys.modules, "torch", fake)
        monkeypatch.setattr(service_module, "torch", fake)


class RecordingPublisher:
    """Capture publish invocations from the perception service."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def publish(self, **kwargs):
        self.calls.append(kwargs)
        return {"seq": 1}


class RecordingTextWorker:
    """Deterministic worker returning predictable embeddings and spans."""

    def __init__(self) -> None:
        self.calls: list[Sequence[tuple[str, float, float]]] = []

    def __call__(self, tokens: Sequence[tuple[str, float, float]], memmap_path: str):
        self.calls.append(tokens)
        features = np.array(
            [[float(i + 1), float(i + 2)] for i in range(len(tokens))], dtype=np.float32
        )
        path = str(memmap_path)
        mm = np.memmap(path, dtype="float32", mode="w+", shape=features.shape)
        mm[:] = features
        mm.flush()
        times = np.array([[start, end] for _, start, end in tokens], dtype=np.float32)
        return mm, times


@pytest.mark.asyncio
async def test_listener_processes_text_only_payload(monkeypatch, tmp_path):
    """INPUT_RECEIVED events with only user_input should yield text spans."""

    _ensure_lightweight_torch(monkeypatch)

    hop = 0.05
    monkeypatch.setattr(
        "deepthought.services.perception.service.get_settings",
        lambda: SimpleNamespace(
            wandb_enabled=False,
            wandb_project=None,
            wandb_sweep_id=None,
            wandb_upload_artifacts=False,
        ),
    )
    service_cfg = SimpleNamespace(
        grid_hop_size=None,
        text_cache_dir=str(tmp_path),
        audio_cache_dir=None,
        video_cache_dir=None,
        audio_window_size=0.02,
        audio_hop_size=0.01,
        audio_model="dummy",
        audio_model_path=None,
        text_model="dummy",
        text_hop_size=hop,
        video_model="dummy",
        video_hop_size=1.0,
    )
    monkeypatch.setattr(
        "deepthought.services.perception.service.PerceptionConfig",
        lambda: service_cfg,
    )
    listener_cfg = SimpleNamespace(enable_asr_transcription=False, text_hop_size=hop)
    monkeypatch.setattr(
        "deepthought.services.perception.listener.PerceptionConfig",
        lambda: listener_cfg,
    )

    publisher = RecordingPublisher()
    worker = RecordingTextWorker()
    service = PerceptionService(publisher, text_worker=worker)
    listener = PerceptionServiceListener(
        service,
        FakeNATS(),
        object(),
        default_user_id="fallback-user",
    )

    payload = {"user_input": "hello world"}
    msg = SimpleNamespace(data=json.dumps(payload).encode(), ack=AsyncMock(), headers=None)

    await listener._handle(msg)

    msg.ack.assert_awaited_once()
    assert len(worker.calls) == 1
    tokens = worker.calls[0]
    assert tokens[0][0] == "hello"
    assert tokens[0][1] == pytest.approx(0.0)
    assert tokens[0][2] == pytest.approx(hop)
    assert tokens[1][0] == "world"
    assert tokens[1][1] == pytest.approx(hop)
    assert tokens[1][2] == pytest.approx(hop * 2)

    assert len(publisher.calls) == 1
    event_kwargs = publisher.calls[0]
    assert event_kwargs["message_id"] == "unknown"
    assert event_kwargs["user_id"] == "fallback-user"

    text_payload = event_kwargs["by_modality"]["text"]
    hop_ms = int(hop * 1000)
    assert text_payload["spans"] == [[0, hop_ms], [hop_ms, hop_ms * 2]]
    assert text_payload["embeddings"] == [[1.0, 2.0], [2.0, 3.0]]
    assert len(text_payload["encoders"]) == 2
    encoder = text_payload["encoders"][0]
    assert encoder["name"] == "RecordingTextWorker"
    assert encoder["modality"] == "text"
    assert encoder["parameters"]["config_source"] == "PerceptionConfig.text_model"
    assert "hop_size" in encoder["parameters"]
    assert event_kwargs["provenance"]["modalities"] == ["text"]
    assert "timestamp" in event_kwargs["provenance"]

