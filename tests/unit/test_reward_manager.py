import asyncio
import types
import numpy as np
import pytest

from deepthought.motivate import reward_manager as rm_mod


class DummyModel:
    def encode(self, text: str, convert_to_numpy: bool = True) -> np.ndarray:
        return np.array([len(text)], dtype=float)


def fake_cos_sim(vec: np.ndarray, arr: np.ndarray) -> np.ndarray:
    arr = np.atleast_2d(arr)
    sims = [1.0 if float(v[0]) == float(vec[0]) else 0.0 for v in arr]
    return np.array([sims], dtype=float)


class DummyResp:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self) -> dict:
        return self._payload


class DummySession:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, *args, **kwargs):
        return DummyResp(self.payload, self.status)


def _create_manager(monkeypatch, payload=None, status=200):
    mgr = rm_mod.RewardManager(None, None, "token")
    monkeypatch.setattr(mgr, "_model", DummyModel())
    monkeypatch.setattr(rm_mod, "util", types.SimpleNamespace(cos_sim=fake_cos_sim))
    monkeypatch.setattr(rm_mod.aiohttp, "ClientSession", lambda: DummySession(payload or {}, status))
    return mgr


def test_score_novelty(monkeypatch):
    mgr = _create_manager(monkeypatch)
    first = mgr._score_novelty("hello")
    repeat = mgr._score_novelty("hello")
    novel = mgr._score_novelty("hi")
    assert first == 1.0
    assert repeat == 0.0
    assert novel == 1.0


@pytest.mark.asyncio
async def test_score_social(monkeypatch):
    mgr = _create_manager(monkeypatch, {"reactions": [{"count": 2}, {"count": 3}]})
    score = await mgr._score_social(123, 456)
    assert score == 5
