import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import types
import pytest

# Provide a lightweight stub of the social_graph_bot module. This allows tests
# to run without installing optional heavy dependencies used by the full
# example implementation.
sg_stub = types.ModuleType("examples.social_graph_bot")

async def _noop(*args, **kwargs):
    return None

sg_stub.send_to_prism = _noop
sg_stub.publish_input_received = _noop

sys.modules.setdefault("examples.social_graph_bot", sg_stub)

# Stub out the optional ``deepthought.motivate`` package to avoid importing
# heavyweight dependencies like sentence-transformers during test collection.
motivate_stub = types.ModuleType("deepthought.motivate")
sys.modules.setdefault("deepthought.motivate", motivate_stub)

import examples.social_graph_bot as sg


@pytest.fixture
def prism_calls(monkeypatch):
    calls = []

    async def fake_send(data):
        calls.append(data)

    monkeypatch.setattr(sg, "send_to_prism", fake_send)
    return calls


@pytest.fixture
def input_events(monkeypatch):
    calls = []

    async def fake_publish(text):
        calls.append(text)

    monkeypatch.setattr(sg, "publish_input_received", fake_publish)
    return calls
