import importlib
import json
import sys
import types

import pytest


class DummyMsg:
    def __init__(self, payload: dict | str):
        if isinstance(payload, dict):
            payload = json.dumps(payload)
        self.data = payload.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


def create_llm(monkeypatch: pytest.MonkeyPatch):
    torch_mod = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc, tb):
            pass

    def no_grad():
        return _NoGrad()

    torch_mod.no_grad = no_grad
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    import deepthought.modules.llm_base as llm_base

    importlib.reload(llm_base)

    class PatchedLLM(llm_base.BaseLLM):
        async def start_listening(self, durable_name: str = "llm") -> bool:  # pragma: no cover - unused
            return True

        async def stop_listening(self) -> None:  # pragma: no cover - unused
            pass

    return PatchedLLM(None, None, None, None, reward_buffer_size=5)


def test_build_prompt_with_rewards(monkeypatch: pytest.MonkeyPatch):
    llm = create_llm(monkeypatch)
    llm._recent_rewards.extend([1.0, 2.0])
    prompt = llm._build_prompt(["fact1", "fact2"])
    assert prompt == "[avg_reward: 1.50]\nfact1\nfact2\nResponse:"


@pytest.mark.asyncio
async def test_handle_reward_event_appends_and_acks(monkeypatch: pytest.MonkeyPatch):
    llm = create_llm(monkeypatch)
    msg = DummyMsg({"reward": 0.8})
    await llm._handle_reward_event(msg)
    assert list(llm._recent_rewards) == [0.8]
    assert msg.acked
