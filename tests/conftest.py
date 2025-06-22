import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

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
